from datetime import timezone as dt_timezone
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

def _get_current_utc_iso() -> str:
    """Returns current UTC timestamp in ISO-8601 format."""
    return datetime.now(dt_timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

class HealthResponse(BaseModel):
    status: str = Field(default='healthy', description='Service health status')
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
    weights: ExplainabilityWeights = Field(default_factory=ExplainabilityWeights, description='Active weighting coefficients')

class EarthIntelligenceResponse(BaseModel):
    timestamp: str = Field(default_factory=_get_current_utc_iso, description='Execution timestamp')
    status: str = Field(default='healthy', description='Computation status')
    ground_elevation_m: float = Field(default=34.6, description='Ground elevation from DEM in meters')
    surface_elevation_m: float = Field(default=43.9, description='Surface elevation from DSM in meters')
    estimated_structure_height_m: float = Field(default=9.3, description='Estimated structure height (Surface - Ground)')
    earth_presence_score: float = Field(default=84.5, description='Normalized Earth Presence Index (0-100)')
    flood_risk_score: float = Field(default=0.92, description='Normalized flood susceptibility risk score (0-1)')
    terrain_suitability_index: float = Field(default=78.4, description='Terrain Suitability Index (0-100)')
    classification: str = Field(default='OPTIMAL', description='Terrain / Presence state classification')
    explainability: ExplainabilityDetails = Field(default_factory=ExplainabilityDetails, description='Explainable AI breakdown')

class SpatialH3TensorRequest(BaseModel):
    lat: float = Field(default=2.0442, ge=-90.0, le=90.0, description='Latitude clamped to [-90, 90]')
    lon: float = Field(default=102.5689, ge=-180.0, le=180.0, description='Longitude clamped to [-180, 180]')
    resolution: int = Field(default=8, ge=7, le=9, description='H3 resolution (7 to 9)')

class SpatialH3TensorResponse(BaseModel):
    h3_index: str = Field(default='886520d81dfffff', description='Uber H3 spatial index string (15 chars)')
    spatial_cell: str = Field(default='886520d81dfffff', description='Duplicate spatial cell reference')
    density_score: float = Field(default=64.2, description='Spatial density score (0-100)')
    earth_presence_index: float = Field(default=73.6, description='Earth Presence Index for cell (0-100)')
    terrain_state_index: float = Field(default=75.1, description='Terrain state index (0-100)')
    flood_state_index: float = Field(default=22.4, description='Flood state index (0-100)')
    status: str = Field(default='ACTIVE', description='Spatial cell operational classification')

class FloodIntelligenceRequest(BaseModel):
    lat: float = Field(default=2.0442, ge=-90.0, le=90.0, description='Latitude clamped to [-90, 90]')
    lon: float = Field(default=102.5689, ge=-180.0, le=180.0, description='Longitude clamped to [-180, 180]')
    radius_km: float = Field(default=5.0, ge=0.1, le=100.0, description='Hydrological basin neighborhood radius')

class FloodIntelligenceResponse(BaseModel):
    timestamp: str = Field(default_factory=_get_current_utc_iso, description='Execution timestamp')
    status: str = Field(default='healthy', description='Computation status')
    twi: float = Field(default=12.8, description='Topographic Wetness Index (TWI)')
    flow_accumulation: int = Field(default=247, description='Upstream flow accumulation cell count')
    slope_percent: float = Field(default=2.4, description='Terrain slope percentage')
    flood_risk_score: float = Field(default=81.3, description='Flood risk score (0-100 scale)')
    classification: str = Field(default='CRITICAL', description='Flood susceptibility classification (LOW, MODERATE, HIGH, CRITICAL)')