import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageOps

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'Nimbus Roman No9 L', 'serif'],
    'font.size': 12
})

image_paths = {
    "28-Day Mortality (XGBoost)": "SHAP/28d_noweight/shap_28d_15_xgb.tiff",
    "90-Day Mortality (XGBoost)": "SHAP/90d_noweight/shap_90d_15_xgb.tiff",
    "1-Year Mortality (XGBoost)": "SHAP/1y_noweight/shap_1y_15_xgb.tiff"
}

def process_img(path):
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    bg = Image.new(img.mode, img.size, (255, 255, 255))
    diff = ImageOps.invert(ImageOps.equalize(ImageOps.grayscale(img)))
    bbox = diff.getbbox()
    if bbox:
        return img.crop(bbox)
    return img

images = {title: process_img(path) for title, path in image_paths.items()}

fig = plt.figure(figsize=(7.5, 8.75))

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.1, wspace=0.05)

axes = [
    fig.add_subplot(gs[0, 0:2]),
    fig.add_subplot(gs[0, 2:4]),
    fig.add_subplot(gs[1, 1:3])
]

for i, (ax, (title, img)) in enumerate(zip(axes, images.items())):
    ax.imshow(img, aspect='equal')
    ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight='bold', pad=5)
    
    ax.text(-0.01, 1.10, chr(65 + i), transform=ax.transAxes, 
            fontsize=10, fontweight='bold', va='top', ha='right')

plt.tight_layout(pad=0.1)
plt.savefig("SHAP/combined_xgb_mortality_noweight.tiff", dpi=300, bbox_inches='tight')
plt.close()
