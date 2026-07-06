# MemDialogue provenance and use conditions

MemDialogue contains model-generated memory-event records produced from the
public, non-toxic WildChat-4.8M release. It contains source conversation hashes for
provenance but does not redistribute the original conversation turns.

The source dataset is WildChat-4.8M:
https://huggingface.co/datasets/allenai/WildChat-4.8M

The release build pins source revision
`c827c6df8fcf008219ffaffa4d1dd77491099367`. WildChat-4.8M is made available
under the Open Data Commons Attribution License (ODC-By) 1.0. MemDialogue is a
derived database distributed under the same ODC-By 1.0 license. Attribution:
"MemDialogue is derived from WildChat-4.8M by the Allen Institute for AI."
The repository's MIT License applies to code only and does not replace the
data license.

The construction pipeline rejects source turns marked toxic, direct
identifiers, sensitive categories, prompt-injection content, and records that
fail an independent release-verification pass. Residual sensitive or unsafe
content may nevertheless remain. Do not attempt to identify users or infer
sensitive personal data. Report questionable records through the repository
issue tracker so they can be reviewed and removed from future releases.

Rebuilding requires downloading WildChat-4.8M directly from its publisher and
retaining the pinned revision, filter settings, model identifier, prompt
version, and rejection statistics in the generated metadata file.
