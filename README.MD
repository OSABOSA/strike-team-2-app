
1. Add full path to main directory (to strike-team-2-app) to custom_path.pth file (create file in ```path_to_venv/lib/pythonX.X/site-packages/custom_path.pth```)
2. Run ```src/data/dataset_downloader.py``` to download dataset
3. Run ```src/data/fix_data.py``` to fix dataset structure (required for pandas, errors otherwise)
4. Run ```src/data/generate_embeddings.py``` to generate embeddings to ```strike-team-2-app/resources/embeddings```