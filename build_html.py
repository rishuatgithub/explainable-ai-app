import json
import os


def build():
    # Read the files
    app_path = "app.py"
    csv_path = os.path.join("data", "heart.csv")
    output_path = "index.html"

    if not os.path.exists(app_path):
        print(f"Error: {app_path} not found.")
        return

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    with open(app_path, "r", encoding="utf-8") as f:
        app_content = f.read()

    with open(csv_path, "r", encoding="utf-8") as f:
        data_content = f.read()

    html_content = f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
    <title>Explaining Heart Diseases using ML Model</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.63.1/build/stlite.css" />
    <style>
      /* Optional: Add spinner styling or other enhancements */
      #loading {{
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-family: sans-serif;
        text-align: center;
        color: #333;
      }}
      .spinner {{
        border: 4px solid rgba(0,0,0,.1);
        width: 36px;
        height: 36px;
        border-radius: 50%;
        border-left-color: #09f;
        animation: spin 1s linear infinite;
        margin: 0 auto 10px auto;
      }}
      @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
      }}
    </style>
  </head>
  <body>
    <div id="loading">
      <div class="spinner"></div>
      <p>Loading WebAssembly Environment...<br><small>This may take a few seconds on first load.</small></p>
    </div>
    <div id="root"></div>
    <script src="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.63.1/build/stlite.js"></script>
    <script>
      stlite.mount(
        {{
          requirements: ["pandas", "scikit-learn", "xgboost", "shap", "eli5", "matplotlib", "plotly", "jinja2"],
          entrypoint: "app.py",
          files: {{
            "app.py": {json.dumps(app_content)},
            "data/heart.csv": {json.dumps(data_content)}
          }},
        }},
        document.getElementById("root")
      ).then(() => {{
        // Remove loading overlay once stlite has loaded the app
        const loadingDiv = document.getElementById("loading");
        if (loadingDiv) {{
          loadingDiv.style.display = "none";
        }}
      }});
    </script>
  </body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print("Successfully generated index.html!")


if __name__ == "__main__":
    build()
