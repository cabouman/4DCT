import os
import mbirjax as mj
import mbirjax.preprocess as mjp
import numpy as np
import time

from utils_multi_threads import mace4d_from_cone_beam_params, truncate_sino_into_time_bins

if __name__ == "__main__":

    output_path = "./output"
    os.makedirs(output_path, exist_ok=True)

    # Set to True if you want to use the saved initial recons
    USE_SAVED_INIT_IMAGE = False

    if USE_SAVED_INIT_IMAGE:
        # Specify init recon path
        init_image_path = ""
        init_image = np.load(init_image_path)
    else:
        init_image = None

    # 4D phantom dataset
    dataset_url = "/depot/bouman/data/Lilly/4DCT/Phantom_30s_Run1_Dec2024.tgz"
    download_dir = "./data"
    os.makedirs(download_dir, exist_ok=True)

    dataset_dir = mj.download_and_extract(dataset_url, download_dir)

    # Preprocessing parameters
    downsample_rate = [1, 1]
    subsample_view_factor = 1
    sino_auto_cropping = True

    # 4D split parameters
    views_per_bin = 48
    stride = 24

    # Recon parameters that must be set on ct_model before truncating into per-bin models.
    sharpness = 1.0
    verbose = 1

    print("\n************** NSI dataset preprocessing **************")
    sino, ct_model = \
        mjp.nsi.get_sino_and_model(dataset_dir, downsample_factor=downsample_rate,
                                   subsample_view_factor=subsample_view_factor, auto_crop=sino_auto_cropping)
    ct_model.set_params(sharpness=sharpness, verbose=verbose, positivity_flag=True)


    print("\n************** Split into time bins **************")
    # Adjust if you want to use a different time range
    start = 0
    end = -1
    time_range = slice(start, end)

    bins = truncate_sino_into_time_bins(
        sino=sino,
        model=ct_model,
        views_per_bin=views_per_bin,
        stride=stride,
    )[time_range]

    print(f"Total bins: {len(bins)}")

    sino_list = []
    model_list = []

    print("\n***************** Reconstruct each bin ****************")
    for t, (sino_t, model, sl) in enumerate(bins):
        sino_list.append(sino_t)
        model_list.append(model)

    # MACE and recon parameters
    weight_type = "transmission_root"
    prior_weight = 0.5
    max_mace_itr = 10
    rho = 0.5
    forward_num_iterations = 3
    stop_threshold = 0.02
    sigma_p = None
    init_save_dir = os.path.join(output_path, "init")
    timing_log_path = os.path.join(output_path, "timing_log.csv")

    time0 = time.time()

    recon_4d = mace4d_from_cone_beam_params(
        sino_list,
        model_list,
        init_image=init_image,
        weight_type=weight_type,
        prior_weight=prior_weight,
        max_mace_itr=max_mace_itr,
        rho=rho,
        forward_num_iterations=forward_num_iterations,
        stop_threshold=stop_threshold,
        sigma_p=sigma_p,
        verbose=verbose,
        init_save_dir=init_save_dir,
        timing_log_path=timing_log_path,
    )

    time1 = time.time()
    run_time = (time1 - time0) / 60 / 60

    np.save(os.path.join(output_path, f"recon_4d_{run_time:.2f}h.npy"), recon_4d)
    print(f"\n[MACE] Total wall time: {run_time:.2f} hours. Saved to {output_path}")