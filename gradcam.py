import tensorflow as tf
import numpy as np
import cv2


# ✅ Get base CNN model (MobileNetV2)
def get_base_model(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("Base CNN model not found.")


# ✅ Grad-CAM
def make_gradcam_heatmap(img_array, model):
    base_model = get_base_model(model)

    # Last conv layer from MobileNetV2
    last_conv_layer = None
    for layer in reversed(base_model.layers):
        if "conv" in layer.name.lower():
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("No conv layer found")

    # 🔥 Create new model ONLY for CNN part
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # Use mean activation (since no final classifier here)
        loss = tf.reduce_mean(predictions)

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)

    if max_val == 0:
        return np.zeros_like(heatmap.numpy())

    heatmap /= max_val

    return heatmap.numpy()


# ✅ Overlay
def overlay_heatmap(heatmap, img, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)

    return cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)