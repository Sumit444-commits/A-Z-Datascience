from flask import Flask, render_template
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt

import io
import base64

app = Flask(__name__)

# Load the dataset
df = sns.load_dataset("iris")

def plot_histogram():
    plt.figure(figsize=(6, 4))
    df.drop(columns=["species"]).hist(figsize=(8, 6), bins=20)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def plot_pairplot():
    plt.figure(figsize=(6, 4))
    sns.pairplot(df, hue="species")

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route("/")
def home():
    summary_stats = df.describe().to_html()

    hist_img = plot_histogram()
    pairplot_img = plot_pairplot()

    return render_template("index.html", summary_stats=summary_stats, hist_img=hist_img, pairplot_img=pairplot_img)

if __name__ == "__main__":
    app.run(debug=True)
