import collections

import numpy as np
import tensorflow as tf
import torch
import timm

from chambers.models.backbones.vision_transformer import ViTB16, ViTB32, ViTL32, ViTL16, DeiTB16
from chambers import augmentations
from chambers.utils.generic import url_to_img
from vit_keras import utils


def pth_to_npz(pth_path):
    pth_contents = torch.load(pth_path)
    if not isinstance(pth_contents, collections.OrderedDict):
        pth_contents = pth_contents["model"]

    npz_contents = {key: tensor.numpy() for key, tensor in pth_contents.items()}
    file_name = os.path.splitext(pth_path)[0] + ".npz"
    np.savez(file_name, **npz_contents)


def map_encoder_layer(l, idx, wdict):
    nh = l.multi_head_attention.num_heads
    hdim = l.multi_head_attention.head_dim
    dim = nh * hdim

    block_name = "blocks." + str(idx)

    wq, wk, wv = (
        wdict[block_name + ".attn.qkv.weight"]
        .reshape([3, nh, hdim, dim])
        .transpose([0, 3, 1, 2])
    )
    bq, bk, bv = wdict[block_name + ".attn.qkv.bias"].reshape([3, nh, 1, hdim])

    wp = (
        wdict[block_name + ".attn.proj.weight"]
        .reshape([dim, nh, hdim])
        .transpose([1, 0, 2])
    )
    bp = np.expand_dims(wdict[block_name + ".attn.proj.bias"], 0)

    wmap_l = {
        l.multi_head_attention: [
            wq,
            bq,
            wv,
            bv,
            wk,
            bk,
            wp,
            bp,
        ],
        l.norm1: [
            wdict[block_name + ".norm1.weight"],
            wdict[block_name + ".norm1.bias"],
        ],
        l.dense1: [
            wdict[block_name + ".mlp.fc1.weight"].transpose(),
            wdict[block_name + ".mlp.fc1.bias"],
        ],
        l.dense2: [
            wdict[block_name + ".mlp.fc2.weight"].transpose(),
            wdict[block_name + ".mlp.fc2.bias"],
        ],
        l.norm2: [
            wdict[block_name + ".norm2.weight"],
            wdict[block_name + ".norm2.bias"],
        ],
    }
    return wmap_l


def set_numpy_weights(model, wdict):
    wmap = {
        # patch embeddings
        model.get_layer("patch_embeddings").get_layer("embedding"): [
            wdict["patch_embed.proj.weight"].transpose([2, 3, 1, 0]),
            wdict["patch_embed.proj.bias"],
        ],
        # cls token
        model.get_layer("add_cls_token"): [wdict["cls_token"][0]],
        # positional embeddings
        model.get_layer("pos_embedding"): [wdict["pos_embed"][0]],
        # transformer output norm
        model.get_layer("encoder").norm_layer: [
            wdict["norm.weight"],
            wdict["norm.bias"],
        ],
    }
    # distillation token
    try:
        map_ = {
            model.get_layer("add_dist_token"): [wdict["dist_token"][0]]
        }
        wmap.update(map_)
    except ValueError:
        # distillation token does not exist
        pass

    # feature/representation
    try:
        map_ = {
            model.get_layer("feature"): [
                wdict["pre_logits.fc.weight"].transpose(),
                wdict["pre_logits.fc.bias"],
            ]
        }
        wmap.update(map_)
    except ValueError:
        # feature layer does not exist
        pass

    # prediction head
    try:
        map_ = {
            model.get_layer("predictions"): [
                wdict["head.weight"].transpose(),
                wdict["head.bias"],
            ]
        }
        wmap.update(map_)
    except ValueError:
        # if "prediction" layer does not exist, assume model was built with include_top=False
        pass

    # prediction head distillation
    try:
        map_ = {
            model.get_layer("predictions_dist"): [
                wdict["head_dist.weight"].transpose(),
                wdict["head_dist.bias"],
            ]
        }
        wmap.update(map_)
    except ValueError:
        # distillation head does not exist
        pass

    # transformer layers
    lmaps = [
        map_encoder_layer(l, idx, wdict)
        for idx, l in enumerate(model.get_layer("encoder").layers)
    ]

    for lmap in lmaps:
        wmap.update(lmap)

    for layer, weights in wmap.items():
        layer.set_weights(weights)


def set_pytorch_weights(model, weights):
    np_weights = {key: tensor.numpy() for key, tensor in weights.items()}
    set_numpy_weights(model, np_weights)


def load_numpy_weights(model, weights_path):
    wdict = np.load(weights_path, allow_pickle=False)
    wdict = dict(wdict)
    set_numpy_weights(model, wdict)


include_top = True
# include_top = False

# model_name = "vit_base_patch16_224"
# model_name = "vit_base_patch16_384"
# model_name = "vit_base_patch32_384"
# model_name = "vit_large_patch16_224"
# model_name = "vit_large_patch16_384"
# model_name = "vit_large_patch32_224"
# model_name = "vit_large_patch32_384"

# model_name = "vit_base_patch16_224_in21k"
# model_name = "vit_large_patch16_224_in21k"
# model_name = "vit_large_patch32_224_in21k"

# model_name = "vit_deit_base_patch16_224"
# model_name = "vit_deit_base_patch16_384"

# model_name = "vit_deit_base_distilled_patch16_224"
model_name = "vit_deit_base_distilled_patch16_384"

pm = timm.create_model(model_name, pretrained=True)

in21k = model_name.endswith("in21k")
img_size = int(model_name.rstrip("_in21k")[-3:])
deit = model_name.lstrip("vit_").startswith("deit")
distilled = "distilled" in model_name

include_top = False if in21k else include_top

sub_name = model_name.lstrip("vit_").lstrip("deit_")
size = sub_name.split("_")[0]

if distilled:
    patch_size = int(sub_name.split("_")[2].lstrip("patch"))
else:
    patch_size = int(sub_name.split("_")[1].lstrip("patch"))

if hasattr(pm.pre_logits, "fc"):
    feature_dim = pm.pre_logits.fc.out_features
else:
    feature_dim = None

if size == "base" and patch_size == 16 and distilled:
    model = DeiTB16
elif size == "base" and patch_size == 16:
    model = ViTB16
elif size == "base" and patch_size == 32:
    model = ViTB32
elif size == "large" and patch_size == 16:
    model = ViTL16
elif size == "large" and patch_size == 32:
    model = ViTL32

if distilled:
    tm = model(input_shape=(img_size, img_size, 3), include_top=include_top, classes=1000)
else:
    tm = model(input_shape=(img_size, img_size, 3), include_top=include_top, classes=1000, feature_dim=feature_dim)

weights = pm.state_dict()
set_pytorch_weights(tm, weights)

# pm
tm.summary()

l = tm.get_layer("encoder").layers[0]
nh = l.multi_head_attention.num_heads
hdim = l.multi_head_attention.head_dim
dim = nh * hdim

batch_size = 2
some_seq_len = 128

# %%
def check_layers(l, l_idx):
    pl = pm.blocks[l_idx]

    x = np.random.uniform(size=(batch_size, some_seq_len, dim)).astype(np.float32)
    xp = torch.tensor(x)

    pz = pl.norm1(xp).detach().numpy()
    tz = l.norm1(x).numpy()
    assert np.allclose(pz, tz, atol=1.0e-5), "norm1"

    pz = pl.norm2(xp).detach().numpy()
    tz = l.norm2(x).numpy()
    assert np.allclose(pz, tz, atol=1.0e-5), "norm2"

    pz = pl.attn(xp).detach().numpy()
    tz = l.multi_head_attention([x, x, x]).numpy()
    assert np.allclose(pz, tz, atol=1.0e-4), "mha"

    pz = pl.mlp(xp).detach().numpy()
    tz = l.dense2(l.dense1(x)).numpy()
    assert np.allclose(pz, tz, atol=1.0e-4), "mlp"

    pz = pl(xp).detach().numpy()
    tz = l(x).numpy()
    assert np.allclose(pz, tz, atol=1.0e-3), "layer"

for idx, l in enumerate(tm.get_layer("encoder").layers):
    check_layers(l, idx)

#%%
x = np.random.uniform(size=(batch_size, img_size, img_size, 3)).astype(np.float32)
cx = x.transpose([0, 3, 1, 2])

pz = pm.patch_embed(torch.tensor(cx)).detach().numpy()
pt = tm.get_layer("patch_embeddings")(x).numpy()

assert np.allclose(pz, pt, atol=1.0e-4)

# %%
if feature_dim is not None:
    pfeature = pm.pre_logits
    tfeature = tm.get_layer("feature")

    x = np.random.uniform(size=(batch_size, dim)).astype(np.float32)
    pz = pfeature(torch.tensor(x)).detach().numpy()
    pt = tfeature(x).numpy()

    assert np.allclose(pz, pt, atol=1.0e-6)

#%%
if include_top:
    phead = pm.head
    thead = tm.get_layer("predictions")

    x = np.random.uniform(size=(batch_size, dim)).astype(np.float32)
    pz = phead(torch.tensor(x)).detach().numpy()
    pt = thead(x).numpy()

    assert np.allclose(pz, pt, atol=1.0e-6)

# %%
if include_top and distilled:
    phead = pm.head_dist
    thead = tm.get_layer("predictions_dist")

    x = np.random.uniform(size=(batch_size, dim)).astype(np.float32)
    pz = phead(torch.tensor(x)).detach().numpy()
    pt = thead(x).numpy()

    assert np.allclose(pz, pt, atol=1.0e-6)

#%%
if include_top:
    url = "https://upload.wikimedia.org/wikipedia/commons/d/d7/Granny_smith_and_cross_section.jpg"
    img = url_to_img(url)
    img = tf.expand_dims(img, 0)

    x = augmentations.Resizing(height=img_size, width=img_size)(img)
    x = augmentations.ImageNetNormalization(mode="tf")(x)

    py = pm(torch.tensor(x.numpy()).permute(0, 3, 1, 2))
    ty = tm.predict(x)
    if distilled:
        py, pyd = py
        pyd = pyd.detach().numpy()
        ty, tyd = ty
        assert np.allclose(pyd, tyd, atol=1.0e-4)

    py = py.detach().numpy()
    assert np.allclose(py, ty, atol=1.0e-3)

    classes = utils.get_imagenet_classes()
    print(classes[py[0].argmax()])  # Granny smith
    print(classes[ty[0].argmax()])  # Granny smith

    # import matplotlib.pyplot as plt
    #
    # fig, axes = plt.subplots(1, 2)
    # axes[0].plot(tf.nn.softmax(py[0]))
    # axes[1].plot(tf.nn.softmax(ty[0]))
    # plt.show()

#%%
import os

save_dir = "keras_weights"

top = "_no_top" if not include_top else ""
deit_prefix = "_deit" if deit else ""
in21k_prefix = "_imagenet21k" if in21k else ""
save_name = "{}{}{}_imagenet_1000_{}{}.h5".format(tm.name, deit_prefix, in21k_prefix, img_size, top)

os.makedirs(save_dir, exist_ok=True)
tm.save_weights(os.path.join(save_dir, save_name))
