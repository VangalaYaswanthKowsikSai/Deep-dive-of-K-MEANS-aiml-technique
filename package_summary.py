
"""
package_summary.py
Generates a simple infographic summarizing the code packages used in the K-Means project.
Usage:
    python package_summary.py
Output:
    - code_package_summary.jpg
"""
import matplotlib.pyplot as plt

packages = [
    ("Python", "Runtime & scripting"),
    ("NumPy", "Numerical arrays & math"),
    ("Pandas", "Data loading & preprocessing"),
    ("scikit-learn", "KMeans, metrics, model API"),
    ("Matplotlib", "Plots & visualizations"),
    ("Joblib", "Model saving & parallelism")
]

def make_summary(outpath='/mnt/data/code_package_summary.jpg'):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    x_start = 0.05
    y = 0.6
    width = 0.27
    height = 0.25
    gap = 0.01

    for i, (name, desc) in enumerate(packages[:3]):
        rect = plt.Rectangle((x_start + i*(width+gap), y), width, height, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x_start + i*(width+gap) + width/2, y + height*0.62, name, ha='center', va='center', fontsize=14, weight='bold')
        ax.text(x_start + i*(width+gap) + width/2, y + height*0.25, desc, ha='center', va='center', fontsize=10)

    y2 = 0.15
    for i, (name, desc) in enumerate(packages[3:]):
        rect = plt.Rectangle((x_start + i*(width+gap), y2), width, height, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x_start + i*(width+gap) + width/2, y2 + height*0.62, name, ha='center', va='center', fontsize=14, weight='bold')
        ax.text(x_start + i*(width+gap) + width/2, y2 + height*0.25, desc, ha='center', va='center', fontsize=10)

    # arrows for flow
    ax.annotate('', xy=(x_start+width/2 + 0.02, y-0.02), xytext=(x_start+width/2 + 0.02, y2+height+0.02),
                arrowprops=dict(arrowstyle='->', linewidth=1.5))
    for i in range(2):
        ax.annotate('', xy=(x_start + (i+0.9)*(width+gap) + width/2, y+height/2), xytext=(x_start + (i+0.1)*(width+gap) + width/2 + width, y+height/2),
                    arrowprops=dict(arrowstyle='->', linewidth=1.2))
    for i in range(2):
        ax.annotate('', xy=(x_start + (i+0.9)*(width+gap) + width/2, y2+height/2), xytext=(x_start + (i+0.1)*(width+gap) + width/2 + width, y2+height/2),
                    arrowprops=dict(arrowstyle='->', linewidth=1.2))

    ax.text(0.5, 0.95, "Code Package Summary — K-Means Project", ha='center', va='center', fontsize=16, weight='bold')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved package summary to {outpath}")

if __name__ == '__main__':
    make_summary('/mnt/data/code_package_summary.jpg')
