"""
Serving feature definitions.

Defines:
- FeatureService: bundles feature views used by the /predict endpoint
- OnDemandFeatureView: optional real-time transformations applied at serving time
"""

from feast import FeatureService

from feast.feature_views.raw_features import raw_event_feature_view

# ── Feature Service ───────────────────────────────────────────────────────────
# The /predict endpoint calls store.get_online_features(feature_service=...).
inference_feature_service = FeatureService(
    name="inference_features",
    features=[raw_event_feature_view],
    description="Features served to the inference endpoint",
)
