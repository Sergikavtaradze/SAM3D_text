import pickle
import argparse
import os

def decode_path(path_data):
    """Decodes a path if it's in bytes format."""
    if isinstance(path_data, bytes):
        try:
            return path_data.decode('utf-8')
        except UnicodeDecodeError:
            print(f"Warning: Could not decode bytes using utf-8: {path_data}")
            # Fallback or handle differently if needed
            return str(path_data)
    return str(path_data) # Ensure it's a string even if not bytes

def read_and_write_paths(pickle_path, output_path):
    """
    Reads a pickle file expected to contain dataset splits with file paths,
    decodes the paths if necessary, and writes them to a text file.
    """
    if not os.path.exists(pickle_path):
        print(f"Error: Pickle file not found at {pickle_path}")
        return

    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
    except pickle.UnpicklingError:
        print(f"Error: Could not unpickle file {pickle_path}. It might be corrupted or not a pickle file.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the pickle file: {e}")
        return

    if not isinstance(data, dict) or not all(k in data for k in ['train', 'val', 'test']):
        print("Warning: Pickle file does not seem to contain the expected structure (dict with 'train', 'val', 'test' keys).")
        # Attempt to process common list structures if dict check fails
        if isinstance(data, list):
             print("Data is a list. Attempting to process elements.")
             # You might need specific logic here based on list content
        elif not isinstance(data, dict):
             print("Cannot determine data structure. Aborting.")
             return # Or implement different handling logic

    try:
        with open(output_path, 'w') as out_f:
            if isinstance(data, dict):
                for split_name, split_data in data.items():
                    out_f.write(f"--- {split_name.upper()} ---\n")
                    if isinstance(split_data, list):
                        for item in split_data:
                            if isinstance(item, dict):
                                if 'image' in item:
                                    img_path = decode_path(item['image'])
                                    out_f.write(f"Image: {img_path}\n")
                                if 'mask' in item:
                                    mask_path = decode_path(item['mask'])
                                    out_f.write(f"Mask:  {mask_path}\n")
                                out_f.write("\n")
                            else:
                                # Handle list items that are not dictionaries
                                out_f.write(f"Item: {decode_path(item)}\n")
                    else:
                        # Handle split data that is not a list
                         out_f.write(f"Data: {decode_path(split_data)}\n")
                    out_f.write("\n")

            elif isinstance(data, list): # Handle if the top-level data is a list
                 out_f.write("--- DATA LIST ---\n")
                 for item in data:
                     # Add logic based on expected item structure in the list
                     out_f.write(f"Item: {decode_path(item)}\n")

            else: # Handle other data types
                out_f.write("--- UNKNOWN DATA STRUCTURE ---\n")
                out_f.write(str(data))

        print(f"Successfully decoded paths written to {output_path}")

    except IOError as e:
        print(f"Error: Could not write to output file {output_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while writing the output file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode paths from a pickle file and write to a text file.")
    parser.add_argument("--pickle_file", help="Path to the input pickle file.")
    parser.add_argument("--output_file", help="Path to the output text file.")

    args = parser.parse_args()

    read_and_write_paths(args.pickle_file, args.output_file)
