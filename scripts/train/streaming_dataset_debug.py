import argparse
import re
import torch
from datasets import load_dataset, IterableDataset, load_from_disk, interleave_datasets

c4_train_path = "/lustre/fsmisc/dataset/HuggingFace/c4/realnewslike/train"
c4_val_path = "/lustre/fsmisc/dataset/HuggingFace/c4/realnewslike/validation"

_sentence_splitter = re.compile(r"(?<=[.!?])\s+")

def sentence_generator(dataset):
    for example in dataset:
        text = example.get("text", "")
        sentences = _sentence_splitter.split(text)
        for sent in sentences:
            sent = sent.strip()
            if sent:
                yield {"text": sent}


def try_shard_and_head(ds, num_shards, idx):
    try:
        sh = ds.shard(num_shards=num_shards, index=idx)
        head = list(sh.take(1))
        return (True, head)
    except Exception as e:
        return (False, repr(e))


def try_interleave(sharded, probs=None, epoch=0):
    try:
        ds = interleave_datasets(sharded, probabilities=probs, seed=42 + epoch)
        it = iter(ds)
        first = next(it)
        return (True, first)
    except Exception as e:
        return (False, repr(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-idx", type=int, default=1)
    args = parser.parse_args()

    print("Loading sources...")
    c4_ds = load_from_disk(c4_train_path).to_iterable_dataset()
    c4_sent = IterableDataset.from_generator(lambda: sentence_generator(c4_ds), features=getattr(c4_ds, "features", None))

    orc_ds = load_dataset("text", data_files="data/orc7.txt", split="train", streaming=True)
    wh_ds = load_dataset("text", data_files="data/wh5.txt", split="train", streaming=True)
    svo_wh_ds = load_dataset("text", data_files="data/declaratives_from_wh5.txt", split="train", streaming=True)
    svo_orc_ds = load_dataset("text", data_files="data/declaratives_from_orc7.txt", split="train", streaming=True)

    sources = [
        ("c4_sent", c4_sent),
        ("orc_ds", orc_ds),
        ("wh_ds", wh_ds),
        ("svo_wh_ds", svo_wh_ds),
        ("svo_orc_ds", svo_orc_ds),
    ]

    print(f"Testing shard behavior with num_shards={args.num_shards}, shard_idx={args.shard_idx}\n")

    for name, ds in sources:
        print(f"-- Source: {name}")
        print(" features:", getattr(ds, "features", None))
        ok, res = try_shard_and_head(ds, args.num_shards, args.shard_idx)
        if ok:
            print("  original shard.take(1) ->", res)
        else:
            print("  original shard() error ->", res)

        # Also test a wrapped version that recreates the iterator per-call
        try:
            if name == "c4_sent":
                wrapped = IterableDataset.from_generator(lambda: sentence_generator(load_from_disk(c4_train_path).to_iterable_dataset()), features=getattr(load_from_disk(c4_train_path).to_iterable_dataset(), "features", None))
            elif name == "orc_ds":
                wrapped = IterableDataset.from_generator(lambda: iter(load_dataset("text", data_files="data/orc7.txt", split="train", streaming=True)), features=None)
            elif name == "wh_ds":
                wrapped = IterableDataset.from_generator(lambda: iter(load_dataset("text", data_files="data/wh5.txt", split="train", streaming=True)), features=None)
            elif name == "svo_wh_ds":
                wrapped = IterableDataset.from_generator(lambda: iter(load_dataset("text", data_files="data/declaratives_from_wh5.txt", split="train", streaming=True)), features=None)
            elif name == "svo_orc_ds":
                wrapped = IterableDataset.from_generator(lambda: iter(load_dataset("text", data_files="data/declaratives_from_orc7.txt", split="train", streaming=True)), features=None)
            else:
                wrapped = ds
            ok2, res2 = try_shard_and_head(wrapped, args.num_shards, args.shard_idx)
            if ok2:
                print("  wrapped shard.take(1) ->", res2)
            else:
                print("  wrapped shard() error ->", res2)
        except Exception as e:
            print("  wrapped construction error ->", repr(e))
        print()

    # Try HF interleave on the per-source shards (simulate in-worker shards)
    sharded_list = []
    probs = [0.9, 0.0, 0.0, 0.05, 0.05]
    for (name, ds), p in zip(sources, probs):
        if not p or p == 0:
            continue
        ok, res = try_shard_and_head(ds, args.num_shards, args.shard_idx)
        if ok:
            try:
                sh = ds.shard(num_shards=args.num_shards, index=args.shard_idx)
            except Exception:
                sh = ds
            sharded_list.append(sh)
        else:
            print(f"Skipping {name} for interleave due to shard error: {res}")

    if sharded_list:
        ok, res = try_interleave(sharded_list, probs=[p for p in probs if p and p > 0], epoch=0)
        if ok:
            print("interleave_datasets first element ->", res)
        else:
            print("interleave_datasets error ->", res)

if __name__ == '__main__':
    main()
