from load_OP import hparams

if hparams['model_size'] == "ViT-B/16":
    alpha_imagenet = 0.6
    theta_imagenet = 0.8
    N_imagenet = 90

    alpha_cub = 0.6
    theta_cub = 0.9
    N_cub = 90

    alpha_pet =0.6
    theta_pet =0.2
    N_pet = 80

    alpha_food =0.6
    theta_food =0.8
    N_food =90

    alpha_place =0.4
    theta_place =0.8
    N_place =60

    alpha_imagenetv2 = 0.5
    theta_imagenetv2 = 0.8
    N_imagenetv2 = 70

    alpha_imagenetr = 0.6
    theta_imagenetr = 0.8
    N_imagenetr = 90

    alpha_imagenets = 0.6
    theta_imagenets = 0.8
    N_imagenets = 80

    alpha_imageneta = 0.5
    theta_imageneta = 0.95
    N_imageneta = 90

if hparams['model_size'] == "ViT-B/32":
    alpha_imagenet = 0.6
    theta_imagenet = 0.8
    N_imagenet = 90

    alpha_cub = 0.5
    theta_cub = 0.95
    N_cub = 80

    alpha_pet =0.6
    theta_pet =0.9
    N_pet = 80

    alpha_food =0.6
    theta_food =0.9
    N_food =80

    alpha_place =0.6
    theta_place =0.9
    N_place =80


if hparams['model_size'] == "ViT-L/14":
    alpha_imagenet = 0.6
    theta_imagenet = 0.8
    N_imagenet = 70

    alpha_cub = 0.5
    theta_cub = 0.9
    N_cub = 80

    alpha_pet =0.6
    theta_pet =0.8
    N_pet = 60

    alpha_food =0.6
    theta_food =0.9
    N_food =70

    alpha_place =0.4
    theta_place =0.9
    N_place =70

