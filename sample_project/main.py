from sample_project.trainer import run_training

if __name__ == "__main__":
    print("[Main] Starting Kairos Labs sample project...")
    model = run_training(epochs=2, batch_size=32)
    print("[Main] Done.")