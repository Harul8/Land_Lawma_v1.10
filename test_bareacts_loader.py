from src.data_loader import load_bare_acts

BAREACTS_FOLDER = "BareActs"

if __name__ == "__main__":
    data = load_bare_acts(BAREACTS_FOLDER)
    print(f"Total Bare Acts loaded: {len(data)}")
