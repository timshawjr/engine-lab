"""Classification gating shared by the retail and smart-city graphs.

EfficientNet-B0 in this project is ImageNet-1k. It has no retail SKU taxonomy
and no surveillance-vehicle taxonomy, so an ungated top-1 label on a small
detector crop is noise, not product identity. Measured on the retail aisle, an
ungated gate produced ``Chihuahua``, ``crash_helmet`` and ``king_penguin`` on a
stationary bowl, and flipped that bowl's label 239 times in 40 seconds.

Three gates make whatever survives defensible:

1. only detector classes the scenario actually cares about are submitted
   (``classify_detector_labels``);
2. the predicted ImageNet class must be plausible *for that detector class*
   (``classify_output_labels``, keyed by detector label);
3. :class:`app.engine.events.EventTracker` additionally requires the same label
   to repeat for ``classify_min_frames`` consecutive frames on one track before
   it is displayed or emitted as an event.

``person`` is hard-banned in code for every scenario: an ImageNet label for a
person crop is always noise, regardless of configuration.

A surviving label is still only an ImageNet candidate. It is never an
authoritative product, SKU, or vehicle identity.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from app.engine.stages import Detection

#: Detector classes that are never submitted to the classifier. Configuration
#: cannot re-enable these; an ImageNet label for a person crop is always noise.
NEVER_CLASSIFY: frozenset[str] = frozenset({"person"})


def classification_candidates(
    detections: tuple[Detection, ...],
    *,
    confidence_min: float,
    top_k: int,
    detector_labels: Iterable[str],
    min_crop_pixels: float = 0.0,
) -> tuple[Detection, ...]:
    """Return the detections allowed to reach the classifier stage.

    ``detector_labels`` is the scenario's ``classify_detector_labels`` rule: the
    detector classes whose crops are meaningful to send to the classifier. This
    is deliberately an allowlist, so an unrecognised or newly added detector
    class fails closed instead of leaking noise into the UI.

    ``min_crop_pixels`` rejects boxes too small to name. A crop is upsampled to
    the classifier's fixed input, so a handful of pixels carries no recoverable
    detail and only produces a confident guess.
    """

    allowed = {value.strip().lower() for value in detector_labels}
    eligible = tuple(
        detection
        for detection in detections
        if detection.confidence >= confidence_min
        and detection.width >= 1.0
        and detection.height >= 1.0
        and min(detection.width, detection.height) >= min_crop_pixels
        and detection.label.lower() not in NEVER_CLASSIFY
        and detection.label.lower() in allowed
    )
    return tuple(sorted(eligible, key=lambda item: item.confidence, reverse=True)[:top_k])


def classification_allowed(
    detector_label: str,
    label: str,
    output_labels: Mapping[str, Iterable[str]],
) -> bool:
    """Return True when a predicted ImageNet class may be shown to the operator.

    ``output_labels`` maps a *detector* class to the ImageNet classes that are
    plausible for it. Scoping per detector class is what stops a ``bowl`` crop
    from matching ``can_opener``: a bowl may only be refined into another
    dishware-like candidate, never into an unrelated kitchen tool.

    A detector class with no entry fails closed.
    """

    permitted = output_labels.get(detector_label.strip().lower())
    if not permitted:
        return False
    allowed = {value.strip().lower() for value in permitted}
    return label.strip().lower() in allowed
