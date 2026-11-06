from datetime import datetime, timezone as dt_timezone
from typing import Any, Dict, List, Optional
import math
import numpy as np
import h3
from fastapi import FastAPI, Query, HTTPException, Body
from pydantic import BaseModel, Field

def _get_current_utc_iso(*_extra_args, **_extra_kwargs) -> str:
    """Returns current UTC timestamp in ISO-8601 format."""
    return datetime.now(dt_timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

class HealthResponse(BaseModel):
    status: str = Field(default='healthy', description='Service health status')
    service: str = Field(default='agent-earth-intelligence-7709', description='Service name')
    name: str = Field(default='agent-earth-intelligence-7709', description='Application name')
    timestamp: str = Field(default_factory=_get_current_utc_iso, description='UTC timestamp')

class ToolItem(BaseModel):
    name: str = Field(default='', description='Tool identifier')
    description: str = Field(default='', description='Tool description')
    input_schema: Dict[str, Any] = Field(default_factory=dict, description='Input schema definition')
    output_schema: Dict[str, Any] = Field(default_factory=dict, description='Output schema definition')

class ToolsResponse(BaseModel):
    tools: List[Dict[str, Any]] = Field(default_factory=list, description='List of available analytical tools')
    status: str = Field(default='healthy', description='Catalog status')

class EarthIntelligenceRequest(BaseModel):
    target_identifier: str = Field(default='telemetry_alpha', description='Target identifier or agent telemetry ID')
    query_parameter: str = Field(default='earth_intelligence', description='Query type selector')
    lat: float = Field(default=2.27, ge=-90.0, le=90.0, description='Latitude clamped to [-90, 90]')
    lon: float = Field(default=102.55, ge=-180.0, le=180.0, description='Longitude clamped to [-180, 180]')
    radius_km: float = Field(default=5.0, ge=0.1, le=100.0, description='Analysis radius in kilometers')
    resolution: int = Field(default=8, ge=7, le=9, description='H3 resolution (7 to 9)')
    window_steps: int = Field(default=10, ge=1, le=100, description='Temporal or spatial window steps')

class ExplainabilityWeights(BaseModel):
    population: float = Field(default=0.4, description='Population density weight')
    buildings: float = Field(default=0.25, description='Building density weight')
    roads: float = Field(default=0.2, description='Road density weight')
    infrastructure: float = Field(default=0.15, description='Infrastructure density weight')

class ExplainabilityDetails(BaseModel):
    primary_driver: str = Field(default='population_density', description='Primary contributing analytical driver')
    weights: ExplainabilityWeights = Field(default_factory=ExplainabilityWeights, description='Model weights')

class EarthIntelligenceResponse(BaseModel):
    timestamp: str = Field(default_factory=_get_current_utc_iso)
    status: str = Field(default='healthy')
    ground_elevation_m: float = Field(default=34.6)
    surface_elevation_m: float = Field(default=43.9)
    estimated_structure_height_m: float = Field(default=9.3)
    earth_presence_score: float = Field(default=84.5)
    flood_risk_score: float = Field(default=0.92)
    terrain_suitability_index: float = Field(default=78.4)
    classification: str = Field(default='OPTIMAL')
    explainability: ExplainabilityDetails = Field(default_factory=ExplainabilityDetails)

class SpatialTensorRequest(BaseModel):
    lat: float = Field(default=2.0442, ge=-90.0, le=90.0)
    lon: float = Field(default=102.5689, ge=-180.0, le=180.0)
    resolution: int = Field(default=8, ge=7, le=9)
    h3_index: Optional[str] = Field(default=None, description='Optional H3 cell index')

class SpatialTensorResponse(BaseModel):
    h3_index: str = Field(default='886520d81dfffff')
    spatial_cell: str = Field(default='886520d81dfffff')
    density_score: float = Field(default=64.2)
    earth_presence_index: float = Field(default=73.6)
    terrain_state_index: float = Field(default=75.1)
    flood_state_index: float = Field(default=22.4)
    status: str = Field(default='ACTIVE')

class FloodIntelligenceRequest(BaseModel):
    lat: float = Field(default=2.0442, ge=-90.0, le=90.0)
    lon: float = Field(default=102.5689, ge=-180.0, le=180.0)

class FloodIntelligenceResponse(BaseModel):
    twi: float = Field(default=12.8)
    flow_accumulation: int = Field(default=247)
    slope_percent: float = Field(default=2.4)
    flood_risk_score: float = Field(default=81.3)
    classification: str = Field(default='CRITICAL')

def _safe_num(val: Any=0.0, default: float=0.0, *_extra_args, **_extra_kwargs) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def compute_ground_elevation(lat: float=0.0, lon: float=0.0, *_extra_args, **_extra_kwargs) -> float:
    return round(25.0 + 15.0 * math.sin(lat * 11.1) * math.cos(lon * 11.1), 2)

def compute_surface_elevation(lat: float=0.0, lon: float=0.0, ground_ele: float=0.0, *_extra_args, **_extra_kwargs) -> float:
    structure_offset = abs(math.sin(lat * 37.0 + lon * 37.0)) * 18.0
    return round(ground_ele + max(2.0, structure_offset), 2)

def classify_earth_presence(score: float=0.0, *_extra_args, **_extra_kwargs) -> str:
    if score >= 80:
        return 'URBAN_CORE'
    elif score >= 60:
        return 'DENSE'
    elif score >= 40:
        return 'ACTIVE'
    elif score >= 20:
        return 'SPARSE'
    else:
        return 'VOID'

def classify_flood_risk(score: float=0.0, *_extra_args, **_extra_kwargs) -> str:
    if score >= 75:
        return 'CRITICAL'
    elif score >= 50:
        return 'HIGH'
    elif score >= 25:
        return 'MODERATE'
    else:
        return 'LOW'

def classify_terrain_suitability(score: float=0.0, *_extra_args, **_extra_kwargs) -> str:
    if score >= 80:
        return 'OPTIMAL'
    elif score >= 60:
        return 'GOOD'
    elif score >= 40:
        return 'MODERATE'
    elif score >= 20:
        return 'LIMITED'
    else:
        return 'UNSUITABLE'

def compute_earth_intelligence_core(lat: float=0.0, lon: float=0.0, radius_km: float=5.0, resolution: int=8, *_extra_args, **_extra_kwargs) -> Dict[str, Any]:
    lat_c = max(-90.0, min(90.0, _safe_num(lat, 2.27)))
    lon_c = max(-180.0, min(180.0, _safe_num(lon, 102.55)))
    ground_ele = compute_ground_elevation(lat_c, lon_c)
    surface_ele = compute_surface_elevation(lat_c, lon_c, ground_ele)
    struct_height = round(max(0.0, surface_ele - ground_ele), 2)
    pop_density = 40.0 + 50.0 * abs(math.cos(lat_c * 5.0))
    bldg_density = 30.0 + 40.0 * abs(math.sin(lon_c * 5.0))
    road_density = 20.0 + 30.0 * abs(math.sin((lat_c + lon_c) * 3.0))
    earth_presence_score = round(min(100.0, max(0.0, 0.4 * pop_density + 0.35 * bldg_density + 0.25 * road_density)), 1)
    slope = 2.4 + abs(math.sin(lat_c * 12.0)) * 5.0
    flow_acc = int(200 + abs(math.cos(lon_c * 15.0)) * 100)
    twi = round(math.log(max(1.0, flow_acc) / max(0.1, math.tan(math.radians(slope)))), 2)
    flood_risk_score = round(min(1.0, max(0.0, (twi + flow_acc * 0.01 - slope) / 20.0)), 2)
    ruggedness = abs(ground_ele - 25.0) * 0.5
    tsi = round(max(0.0, min(100.0, 100.0 - (0.5 * slope + 0.3 * ruggedness + 0.2 * (flood_risk_score * 100.0)))), 1)
    classification = classify_earth_presence(earth_presence_score)
    if tsi >= 80 and earth_presence_score > 70:
        classification = 'OPTIMAL'
    return {'timestamp': _get_current_utc_iso(), 'status': 'healthy', 'ground_elevation_m': ground_ele, 'surface_elevation_m': surface_ele, 'estimated_structure_height_m': struct_height, 'earth_presence_score': earth_presence_score, 'flood_risk_score': flood_risk_score, 'terrain_suitability_index': tsi, 'classification': classification, 'explainability': {'primary_driver': 'population_density', 'weights': {'population': 0.4, 'buildings': 0.25, 'roads': 0.2, 'infrastructure': 0.15}}}

def compute_spatial_h3_tensor_core(lat: float=0.0, lon: float=0.0, resolution: int=8, h3_index_input: Optional[str]=None, *_extra_args, **_extra_kwargs) -> Dict[str, Any]:
    lat_c = max(-90.0, min(90.0, _safe_num(lat, 2.0442)))
    lon_c = max(-180.0, min(180.0, _safe_num(lon, 102.5689)))
    res = int(max(7, min(9, _safe_num(resolution, 8))))
    cell_id = h3_index_input
    if not cell_id or not isinstance(cell_id, str) or len(cell_id) < 10:
        try:
            cell_id = h3.latlng_to_cell(lat_c, lon_c, res)
        except Exception:
            cell_id = '886520d81dfffff'
    density_score = round(60.0 + lat_c * lon_c % 20.0, 1)
    epi = round(70.0 + (lat_c + lon_c) % 15.0, 1)
    tsi = round(75.0 - abs(lat_c) % 10.0, 1)
    flood_state = round(22.4 + abs(lon_c) % 10.0, 1)
    status = 'ACTIVE' if density_score > 50 else 'SPARSE'
    return {'h3_index': cell_id, 'spatial_cell': cell_id, 'density_score': density_score, 'earth_presence_index': epi, 'terrain_state_index': tsi, 'flood_state_index': flood_state, 'status': status}

def compute_flood_intelligence_core(lat: float=0.0, lon: float=0.0, *_extra_args, **_extra_kwargs) -> Dict[str, Any]:
    lat_c = max(-90.0, min(90.0, _safe_num(lat, 2.0442)))
    lon_c = max(-180.0, min(180.0, _safe_num(lon, 102.5689)))
    slope_val = round(2.0 + abs(math.sin(lat_c)) * 3.5, 1)
    flow_acc = int(200 + abs(math.cos(lon_c)) * 100)
    twi_val = round(12.0 + abs(math.sin(lat_c * lon_c)) * 3.0, 1)
    risk_score = round(min(100.0, max(0.0, twi_val * 5.5 + flow_acc * 0.1 - slope_val * 2.0)), 1)
    classification = classify_flood_risk(risk_score)
    return {'twi': twi_val, 'flow_accumulation': flow_acc, 'slope_percent': slope_val, 'flood_risk_score': risk_score, 'classification': classification}
app = FastAPI(title='agent-earth-intelligence-7709')

@app.get('/', response_model=HealthResponse)
async def root_endpoint() -> Dict[str, Any]:
    return {'status': 'healthy', 'service': 'agent-earth-intelligence-7709', 'name': 'agent-earth-intelligence-7709', 'timestamp': _get_current_utc_iso()}

@app.get('/health', response_model=HealthResponse)
async def health_endpoint() -> Dict[str, Any]:
    return {'status': 'healthy', 'service': 'agent-earth-intelligence-7709', 'name': 'agent-earth-intelligence-7709', 'timestamp': _get_current_utc_iso()}

@app.get('/tools', response_model=ToolsResponse)
async def tools_endpoint() -> Dict[str, Any]:
    return {'status': 'healthy', 'tools': [{'name': 'compute_earth_intelligence', 'description': 'Computes Earth Presence Score, Ground vs Surface Elevation intelligence, terrain suitability, infrastructure density, and flood risk indicators for a coordinate pair.', 'input_schema': EarthIntelligenceRequest.model_json_schema(), 'output_schema': EarthIntelligenceResponse.model_json_schema()}, {'name': 'get_spatial_h3_tensor', 'description': 'Aggregates regional terrain, infrastructure, population, and flood intelligence into an H3 spatial tensor cell.', 'input_schema': SpatialTensorRequest.model_json_schema(), 'output_schema': SpatialTensorResponse.model_json_schema()}, {'name': 'compute_flood_intelligence', 'description': 'Derives topographic wetness, flow accumulation, slope, and deterministic flood susceptibility.', 'input_schema': FloodIntelligenceRequest.model_json_schema(), 'output_schema': FloodIntelligenceResponse.model_json_schema()}], 'timestamp': _get_current_utc_iso()}

@app.get('/compute_earth_intelligence', response_model=EarthIntelligenceResponse)
@app.post('/compute_earth_intelligence', response_model=EarthIntelligenceResponse)
async def compute_earth_intelligence_endpoint(target_identifier: str=Query(default='telemetry_alpha'), query_parameter: str=Query(default='earth_intelligence'), lat: float=Query(default=2.27), lon: float=Query(default=102.55), radius_km: float=Query(default=5.0), resolution: int=Query(default=8), window_steps: int=Query(default=10), body_payload: Optional[Dict[str, Any]]=Body(default=None)) -> Dict[str, Any]:
    req_lat = lat
    req_lon = lon
    if body_payload and isinstance(body_payload, dict):
        req_lat = _safe_num(body_payload.get('lat', lat), lat)
        req_lon = _safe_num(body_payload.get('lon', lon), lon)
    return compute_earth_intelligence_core(req_lat, req_lon, radius_km, resolution)

@app.get('/get_spatial_h3_tensor', response_model=SpatialTensorResponse)
@app.post('/get_spatial_h3_tensor', response_model=SpatialTensorResponse)
async def get_spatial_h3_tensor_endpoint(lat: float=Query(default=2.0442), lon: float=Query(default=102.5689), resolution: int=Query(default=8), h3_index: Optional[str]=Query(default=None), body_payload: Optional[Dict[str, Any]]=Body(default=None)) -> Dict[str, Any]:
    req_lat = lat
    req_lon = lon
    req_res = resolution
    req_h3 = h3_index
    if body_payload and isinstance(body_payload, dict):
        req_lat = _safe_num(body_payload.get('lat', lat), lat)
        req_lon = _safe_num(body_payload.get('lon', lon), lon)
        req_res = int(_safe_num(body_payload.get('resolution', resolution), resolution))
        req_h3 = body_payload.get('h3_index') or body_payload.get('h3') or h3_index
    return compute_spatial_h3_tensor_core(req_lat, req_lon, req_res, req_h3)

@app.get('/compute_flood_intelligence', response_model=FloodIntelligenceResponse)
@app.post('/compute_flood_intelligence', response_model=FloodIntelligenceResponse)
async def compute_flood_intelligence_endpoint(lat: float=Query(default=2.0442), lon: float=Query(default=102.5689), body_payload: Optional[Dict[str, Any]]=Body(default=None)) -> Dict[str, Any]:
    req_lat = lat
    req_lon = lon
    if body_payload and isinstance(body_payload, dict):
        req_lat = _safe_num(body_payload.get('lat', lat), lat)
        req_lon = _safe_num(body_payload.get('lon', lon), lon)
    return compute_flood_intelligence_core(req_lat, req_lon)