import os
import urllib
import gzip


def download_embedding():
    url = 'https://zenodo.org/record/3237458/files/glove.twitter.27B.25d.txt.gz?download=1'
    filename = 'glove.twitter.27B.25d.txt'

    if not os.path.exists(filename):
        # Téléchargement du fichier gzip
        print("Download the file...")
        urllib.request.urlretrieve(url, filename + '.gz')

        # Décompression du fichier gzip
        print("Decompressing the file...")
        with gzip.open(filename + '.gz', 'rb') as f_in:
            with open(filename, 'wb') as f_out:
                f_out.write(f_in.read())

        # Suppression du fichier gzip
        print("Deleting the gzip file...")
        os.remove(filename + '.gz')

        print("Download complete.")
    else:
        print("The file already exists.")


if __name__ == "__main__":
    download_embedding()
