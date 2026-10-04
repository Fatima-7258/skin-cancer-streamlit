import os, json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🩺", layout="wide")

@st.cache_resource
def load_artifacts():
    with open(os.path.join(HERE, "metadata.json")) as f:
        meta = json.load(f)
    models = [keras.models.load_model(os.path.join(HERE, fn), compile=False) for fn in meta["keras_files"]]
    return meta, models

meta, models = load_artifacts()
classes, S = meta["class_names"], meta["img_size"]

def prep(pil_img):
    arr = np.asarray(pil_img.convert("RGB"))
    x = tf.image.resize(arr, (S, S), antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(x), 0, 255), tf.uint8).numpy().astype("float32")

def predict(x):                       # x: (S, S, 3) float32 in 0-255
    b = x[None]
    probs = np.mean([m.predict(b, verbose=0) for m in models], axis=0)
    if meta.get("use_tta"):
        probs_f = np.mean([m.predict(b[:, :, ::-1, :], verbose=0) for m in models], axis=0)
        probs = (probs + probs_f) / 2
    return probs[0]

def grad_cam(x):
    model, last = models[0], meta.get("last_conv")
    gm = keras.Model(model.inputs, [model.get_layer(last).output, model.output])
    with tf.GradientTape() as tape:
        conv, preds = gm(x[None], training=False)
        score = tf.gather(preds[0], tf.argmax(preds[0]))
    grads = tape.gradient(score, conv)
    w = tf.reduce_mean(tf.cast(grads, tf.float32), axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(tf.cast(conv, tf.float32) * w[:, None, None, :], axis=-1))[0]
    cam = (cam / (tf.reduce_max(cam) + 1e-8)).numpy()
    heat = tf.image.resize(cam[..., None], (S, S)).numpy()[..., 0]
    colored = (cm.jet(heat)[..., :3] * 255).astype(np.uint8)
    return (x * 0.55 + colored * 0.45).astype(np.uint8)

st.title("🩺 Skin Lesion Classifier")
st.warning("Educational demo only. This is NOT a medical device and must not be used for diagnosis. "
           "Always consult a qualified dermatologist.")
tm = meta["test_metrics"]
extra = f" | Sensitivity: **{tm['sensitivity']:.3f}** | Specificity: **{tm['specificity']:.3f}**" if "sensitivity" in tm else ""
st.caption(f"Model: **{meta['model_name']}** | Test accuracy: **{tm['accuracy']:.3f}** | Test F1 (macro): **{tm['f1_macro']:.3f}**{extra}")

files = st.file_uploader("Upload one or more skin lesion images", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)
if files:
    rows = []
    for up in files:
        img = Image.open(up)
        x = prep(img)
        probs = predict(x)
        top = int(np.argmax(probs))
        rows.append({"file": up.name, "prediction": classes[top], "confidence": round(float(probs[top]), 4)})
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(img.convert("RGB"), caption=up.name, use_container_width=True)
        with c2:
            st.subheader(f"Prediction: {classes[top]}  ({probs[top]:.1%})")
            st.bar_chart(pd.Series(probs, index=classes, name="probability"))
            if meta.get("last_conv"):
                with st.expander("Show Grad-CAM (where the model looked)"):
                    try:
                        st.image(grad_cam(x), use_container_width=True)
                    except Exception as e:
                        st.info(f"Grad-CAM unavailable: {e}")
        st.divider()
    if len(rows) > 1:
        st.dataframe(pd.DataFrame(rows))
