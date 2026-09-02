# Sphinx Documentation Plan — MACE4DModel in mbirjax

The code merge is complete (mbirjax branch `4DCT_for_merging`, commits `ba42ee6`, `89faa06`,
`4686a2d`, `87b784d`).  `MACE4DModel` and the frame utilities are exported and tested, but
they appear nowhere in the rendered documentation: `docs/source/usr_api.rst` has no 4D page,
the API overview does not mention 4D reconstruction, and `usr_parameters.rst` documents no
MACE parameter.  This plan closes that gap.

The goal is that a reader who knows mbirjax but not this work can find 4D reconstruction from
the front page, understand what problem it solves, and run it — without reading `mace4d.py`.

## 1. What Sphinx already gives us for free

The docstrings are Google style and `napoleon` is enabled, so `autoclass`/`automethod` render
`MACE4DModel` and its `recon` correctly with no docstring changes.  `save_volume_as_gif`,
`construct_time_frames` and `construct_time_frame_models` likewise.  The work below is
therefore mostly *placement*: telling Sphinx where these belong and writing the connective
prose that autodoc cannot generate.

Two details of `docs/source/conf.py` that shape the work:

* `autodoc_default_options` sets only `exclude-members: __init__`, and nothing turns on
  `inherited-members`.  So `.. autoclass::` without `:members:` renders the class docstring
  alone — the pattern `usr_denoising.rst` already uses.  `MACE4DModel` being the first
  `ParameterHandler`-only model in the docs therefore raises no inherited-member question;
  the constructor arguments must live in the class docstring, which they already do.
* `conf.py` carries a `nitpick`-style list of unresolvable annotations
  (`('py:class', 'mbirjax.utilities.ObjectType')` and similar).  `MACE4DParamNames` is a
  module-level `Literal` alias used in the `get_params` overload, exactly the shape of the
  entries already listed, so expect to add `('py:class', 'mbirjax.mace4d.MACE4DParamNames')`
  there if the build warns.

## 2. New page: `docs/source/usr_mace4d.rst`

Modeled on `usr_denoising.rst`, which is the closest existing page: one class, a short
statement of what it computes, then the constructor and the primary method.

Structure:

```
.. _MACE4DDocs:

=====================
4D Reconstruction
=====================

<what it is: a time sequence of volumes from one continuous scan>

<the frame decomposition: frames_per_rotation, frame_overlap_factor, and the picture of
 overlapping angular windows -- this is the one concept a reader must hold>

<the MACE structure: one prox_map per frame + three hyperplane qGGMRF denoisers, reconciled
 by consensus equilibrium; cite the MACE reference in refs.bib>

<the dejitter: why gating imprints a periodic modulation on the time axis and why it is
 removed inside every agent>

Constructor
-----------
.. autoclass:: mbirjax.MACE4DModel
   :show-inheritance:

Reconstruction
--------------
.. automethod:: mbirjax.MACE4DModel.recon

Device configuration
--------------------
.. automethod:: mbirjax.MACE4DModel.set_device_pool

Time frames
-----------
.. autofunction:: mbirjax.utilities.construct_time_frame_models
.. autofunction:: mbirjax.utilities.construct_time_frames
```

Two pieces of prose that no docstring covers and that a user will otherwise get wrong:

* **Weighting.**  `weights=None` means unit weights, matching `TomographyModel.recon`, but
  `transmission_root` is the validated setting for 4D transmission data.  State the one-line
  `gen_weights` call in a `code-block`, prominently, so the default is not mistaken for the
  recommendation.
* **Frame count.**  Frames are derived from the model's angle spacing, so `nt` is a
  consequence of `frames_per_rotation`, `frame_overlap_factor` and the scan length, not
  something the user sets.  Show `mace.nt` and `mace.view_slices` as the way to check the
  decomposition before committing to a long run.

Include the complete worked example from the class docstring, extended with the
`slice_viewer`/`save_volume_as_gif` call, so the page ends with something runnable.

## 3. Edits to existing pages

| File | Edit |
|------|------|
| `docs/source/usr_api.rst` | Add `:ref:`MACE4DDocs`` to the bullet list and `usr_mace4d` to the hidden toctree, after `usr_denoising`. |
| `docs/source/usr_api_overview.rst` | New "4D Reconstruction" section after "Denoising", with an `autosummary` for `MACE4DModel.recon` and one sentence on when to use it. |
| `docs/source/usr_utilities.rst` | Add `construct_time_frames` and `construct_time_frame_models` under "General Purpose", next to `copy_ct_model`.  Add `save_volume_as_gif` under the viewer section, with one line saying it steps a 3D volume over its first axis or a 4D volume over time, and lays a plane out the way `slice_viewer` does. |
| `docs/source/usr_parameters.rst` | New "4D MACE Parameters" section documenting `mace_prior_weight`, `rho_mann`, `prox_num_iterations`, `prox_stop_threshold`, `dejitter` in the page's existing `:Type:` format, each with a `.. _param-<name>:` anchor. Note that `sigma_prox` and `verbose` are shared with the base parameters. |
| `docs/source/usr_multi_gpu.rst` | Short subsection: 4D uses one task per device (one worker thread each), not the sharded layout the rest of the page describes, and every per-frame model and denoiser is pinned with `configure_devices([device])`. Say why: auto-sharding inside concurrent threads opens one collective clique per thread and deadlocks. |
| `docs/source/demos_and_faqs.rst` | Link the 4D demo added in section 4. |
| `docs/source/index.rst` | Add 4D reconstruction to the "Key features" list. |

## 4. Demo script: `demo/demo_11_mace4d.py`

The docs pages link to `demo/`, and every other major feature has a numbered demo.  A 4D demo
needs a time-varying phantom, which `generate_demo_data` does not produce, so the demo must:

1. Build a cone-beam model over a full rotation.
2. Build a time-varying phantom — the simplest honest choice is a Shepp-Logan phantom with one
   ellipsoid translating over time, `nt` volumes total.
3. Forward project each time frame with the per-frame models from
   `construct_time_frame_models`, taking each frame's views from the phantom state at that
   time.  This is exactly the use case the model-only primitive exists for, so the demo
   doubles as its motivating example.
4. Reconstruct with `MACE4DModel` and display with `slice_viewer` / `save_volume_as_gif`.

Size it to run on a laptop CPU in a few minutes (small detector, few frames, `max_iterations`
around 3), with a comment on the settings to raise for a GPU.

**Cost note:** this is the largest item in this plan — it is new code, not prose, and needs its
own correctness check that the moving feature actually moves in the reconstruction.  It could
reasonably ship one commit after the docs pages.

## 5. Release notes

Three user-visible changes from this merge belong in the release notes, none of which are
discoverable from the API pages:

* `MACE4DModel`, `construct_time_frames`, `construct_time_frame_models` are new.
* `save_volume_as_gif` now takes a 3D or a 4D volume, dispatching on `ndim` with no new
  arguments: 3D steps over axis 0 as before, 4D plays over time at the middle slice of axis 1.
  Two changes to note in the release notes: the displayed axes are now in ascending order,
  matching `slice_viewer`, where the 3D writer used to transpose them; and `vmin`/`vmax` now
  default to the range of the frames shown rather than to a fixed (0, 1).
* Model loggers are now per instance rather than per class.  Anyone configuring mbirjax
  logging by class name (`logging.getLogger('ConeBeamModel')`) no longer reaches the model's
  logger.  This is the only behavior change that can break an existing setup.

## 6. Build and check

```bash
pip install -e ".[docs]"
cd docs && make clean html
```

Check specifically:

* No new warnings from the added pages (`make html` surfaces broken `:ref:` targets and
  malformed docstrings as warnings; treat any new one as a failure).
* No unresolved-reference warning for `MACE4DParamNames` (section 1); add the `conf.py`
  entry if there is one.
* The `Example:` blocks in the docstrings render as code, not as prose.

## 7. Suggested commit sequence

1. `usr_mace4d.rst` + the `usr_api.rst` toctree entry — the page exists and is reachable.
2. Overview, utilities, parameters, multi-GPU, index edits — cross-links from everywhere a
   reader might start.
3. `demo/demo_11_mace4d.py` + the `demos_and_faqs.rst` link.
4. Release notes.

Steps 1 and 2 are prose over existing docstrings and carry no risk beyond build warnings.
Step 3 is real code and should be run before it is committed.
