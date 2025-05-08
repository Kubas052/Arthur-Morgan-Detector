import os


normal_dir = "../data/TB_Chest_Radiography_Database/Normal"
tb_dir = "../data/TB_Chest_Radiography_Database/Tuberculosis"


# make it a test
def verify_paths(paths):
    print("\n=== Path verification ===")
    for path in paths:
        abs_path = os.path.abspath(path)
        exists = os.path.exists(abs_path)
        is_dir = os.path.isdir(abs_path) if exists else False
        file_count = len(os.listdir(abs_path)) if is_dir else 0

        print(f"\nPath: {abs_path}")
        print(f"Status: {'OK' if exists else 'ERROP'}")
        print(f"File count: {file_count}")
        if is_dir and file_count > 0:
            print("Random files:", os.listdir(abs_path)[:3])
    print("=" * 50)


verify_paths([normal_dir, tb_dir])