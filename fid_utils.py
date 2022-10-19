import gc
import io
import os

import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
import evaluation

def get_fid(config, sampling_fn, score_model, eval_dir='assets/stats', job_name='cifar10'):
    # Use inceptionV3 for images with resolution higher than 256.
    inceptionv3 = config.data.image_size >= 256
    inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

    eval_dir = os.path.join(eval_dir, job_name)
    os.makedirs(eval_dir, exist_ok=True)

    # config.eval.num_samples = 32
    print("num of all generations:", config.eval.num_samples, "eval_batch_size:", config.eval.batch_size)
    num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
    for r in range(num_sampling_rounds):
        print("sampling -- round: %d" % (r))

        samples, n = sampling_fn(score_model)
        samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        samples = samples.reshape(
            (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        # Write samples to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
                os.path.join(eval_dir, f"samples_{r}.npz"), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, samples=samples)
            fout.write(io_buffer.getvalue())

        # Force garbage collection before calling TensorFlow code for Inception network
        gc.collect()
        latents = evaluation.run_inception_distributed(samples, inception_model,
                                                       inceptionv3=inceptionv3)
        # Force garbage collection again before returning to JAX code
        gc.collect()
        # Save latent represents of the Inception network to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
                os.path.join(eval_dir, f"statistics_{r}.npz"), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(
                io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
            fout.write(io_buffer.getvalue())

    # Compute inception scores, FIDs and KIDs.
    # Load all statistics that have been previously computed and saved for each host
    all_logits = []
    all_pools = []
    stats = tf.io.gfile.glob(os.path.join(eval_dir, "statistics_*.npz"))
    for stat_file in stats:
        with tf.io.gfile.GFile(stat_file, "rb") as fin:
            stat = np.load(fin)
            if not inceptionv3:
                all_logits.append(stat["logits"])
            all_pools.append(stat["pool_3"])

    if not inceptionv3:
        all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
    all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

    # Load pre-computed dataset statistics.
    data_stats = evaluation.load_dataset_stats(config)
    data_pools = data_stats["pool_3"]

    # Compute FID/KID/IS on all samples together.
    if not inceptionv3:
        inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
    else:
        inception_score = -1

    fid = tfgan.eval.frechet_classifier_distance_from_activations(
        data_pools, all_pools)
    # Hack to get tfgan KID work for eager execution.
    tf_data_pools = tf.convert_to_tensor(data_pools)
    tf_all_pools = tf.convert_to_tensor(all_pools)
    kid = tfgan.eval.kernel_classifier_distance_from_activations(
        tf_data_pools, tf_all_pools).numpy()
    del tf_data_pools, tf_all_pools

    print(
        " --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
            inception_score, fid, kid))

    with tf.io.gfile.GFile(os.path.join(eval_dir, f"report.npz"), "wb") as f:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
        f.write(io_buffer.getvalue())