#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
import tqdm
import requests

REPOSITORY_PATH = "https://github.com/hse-aml/natural-language-processing"


def download_file(url, file_path):
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length'))
    try:
        with open(file_path, 'wb', buffering=16 * 1024 * 1024) as f:
            bar = tqdm.tqdm_notebook(total=total_size, unit='B', unit_scale=True)
            bar.set_description(os.path.split(file_path)[-1])
            for chunk in r.iter_content(32 * 1024):
                f.write(chunk)
                bar.update(len(chunk))
    except Exception:
        print("Download failed")
    finally:
        if os.path.getsize(file_path) != total_size:
            os.remove(file_path)
            print("Removed incomplete download")


def download_from_github(version, fn, target_dir, force=False):
    url = REPOSITORY_PATH + "/releases/download/{0}/{1}".format(version, fn)
    file_path = os.path.join(target_dir, fn)
    if os.path.exists(file_path) and not force:
        print("File {} is already downloaded.".format(file_path))
        return
    download_file(url, file_path)


def sequential_downloader(version, fns, target_dir, force=False):
    os.makedirs(target_dir, exist_ok=True)
    for fn in fns:
        download_from_github(version, fn, target_dir, force=force)


def link_all_files_from_dir(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for fn in os.listdir(src_dir):
        src_file = os.path.join(src_dir, fn)
        dst_file = os.path.join(dst_dir, fn)
        if os.name == "nt":
            shutil.copyfile(src_file, dst_file)
        else:
            if not os.path.exists(dst_file):
                os.symlink(os.path.abspath(src_file), dst_file)


def link_resources():
    link_all_files_from_dir("../readonly/dataset/", ".")


def download_project_resources(force=False):
    sequential_downloader(
        "project",
        [
            "dialogues.tsv",
            "tagged_posts.tsv",
        ],
        "data",
        force=force
    )
