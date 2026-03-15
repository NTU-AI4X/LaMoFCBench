from .codecs import Elic2022OfficialFeatureCoding, ScaleHyperpriorFeatureCoding

AVAILABLE_CODECS = {
    "elic-featurecoding": Elic2022OfficialFeatureCoding,
    "hyperprior-featurecoding": ScaleHyperpriorFeatureCoding,
}
