import base64


def convert_file_to_b64_string(file_path):
    """
    base64 encoder
    """
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read())


def convert_b64_string_to_file(s, outfile_path):
    """
    base64 decoder
    """
    with open(outfile_path, "wb") as f:
        f.write(base64.b64decode(s))


if __name__ == "__main__":
    encode_files = ['dataset', 'main', 'make_folds', 'models', 'transforms', 'utils']

    for file in encode_files:
        with open(file+".txt", "wb") as f:
            f.write(convert_file_to_b64_string(file+".py"))
