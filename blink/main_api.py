# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import json

import numpy as np
import torch
from colorama import init
from fastapi import FastAPI
from termcolor import colored
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm

import blink.candidate_ranking.utils as utils
import blink.ner as NER
from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder
from blink.biencoder.data_process import (get_candidate_representation,
                                          process_mention_data)
from blink.crossencoder.crossencoder import (CrossEncoderRanker,
                                             load_crossencoder)
from blink.crossencoder.data_process import prepare_crossencoder_data
from blink.crossencoder.train_cross import evaluate, modify
from blink.index.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer

HIGHLIGHTS = [
    "on_red",
    "on_green",
    "on_yellow",
    "on_blue",
    "on_magenta",
    "on_cyan",
]


def _print_colorful_text(input_sentence, samples):
    init()  # colorful output
    msg = ""
    if samples and (len(samples) > 0):
        msg += input_sentence[0 : int(samples[0]["start_pos"])]
        for idx, sample in enumerate(samples):
            msg += colored(
                input_sentence[int(sample["start_pos"]) : int(sample["end_pos"])],
                "grey",
                HIGHLIGHTS[idx % len(HIGHLIGHTS)],
            )
            if idx < len(samples) - 1:
                msg += input_sentence[
                    int(sample["end_pos"]) : int(samples[idx + 1]["start_pos"])
                ]
            else:
                msg += input_sentence[int(sample["end_pos"]) :]
    else:
        msg = input_sentence
    print("\n" + str(msg) + "\n")


def _print_colorful_prediction(
    idx, sample, e_id, e_title, e_text, e_url, show_url=False
):
    print(colored(sample["mention"], "grey", HIGHLIGHTS[idx % len(HIGHLIGHTS)]))
    to_print = "id:{}\ntitle:{}\ntext:{}\n".format(e_id, e_title, e_text[:256])
    if show_url:
        to_print += "url:{}\n".format(e_url)
    print(to_print)


def _annotate(ner_model, input_sentences):
    ner_output_data = ner_model.predict(input_sentences)
    sentences = ner_output_data["sentences"]
    mentions = ner_output_data["mentions"]
    samples = []
    for mention in mentions:
        record = {}
        record["label"] = "unknown"
        record["label_id"] = -1
        # LOWERCASE EVERYTHING !
        record["context_left"] = sentences[mention["sent_idx"]][
            : mention["start_pos"]
        ].lower()
        record["context_right"] = sentences[mention["sent_idx"]][
            mention["end_pos"] :
        ].lower()
        record["mention"] = mention["text"].lower()
        record["start_pos"] = int(mention["start_pos"])
        record["end_pos"] = int(mention["end_pos"])
        record["sent_idx"] = mention["sent_idx"]
        samples.append(record)
    return samples


def _load_candidates(
    entity_catalogue, entity_encoding, faiss_index=None, index_path=None, logger=None
):
    # only load candidate encoding if not using faiss index
    if faiss_index is None:
        candidate_encoding = torch.load(entity_encoding)
        indexer = None
    else:
        if logger:
            logger.info("Using faiss index to retrieve entities.")
        candidate_encoding = None
        assert index_path is not None, "Error! Empty indexer path."
        if faiss_index == "flat":
            indexer = DenseFlatIndexer(1)
        elif faiss_index == "hnsw":
            indexer = DenseHNSWFlatIndexer(1)
        else:
            raise ValueError("Error! Unsupported indexer type! Choose from flat,hnsw.")
        indexer.deserialize_from(index_path)

    # load all the 5903527 entities
    title2id = {}
    id2title = {}
    id2text = {}
    wikipedia_id2local_id = {}
    local_idx = 0
    with open(entity_catalogue, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)

            if "idx" in entity:
                split = entity["idx"].split("curid=")
                if len(split) > 1:
                    wikipedia_id = int(split[-1].strip())
                else:
                    wikipedia_id = entity["idx"].strip()

                assert wikipedia_id not in wikipedia_id2local_id
                wikipedia_id2local_id[wikipedia_id] = local_idx

            title2id[entity["title"]] = local_idx
            id2title[local_idx] = entity["title"]
            id2text[local_idx] = entity["text"]
            local_idx += 1
    return (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        indexer,
    )


def __map_test_entities(test_entities_path, title2id, logger):
    # load the 732859 tac_kbp_ref_know_base entities
    kb2id = {}
    missing_pages = 0
    n = 0
    with open(test_entities_path, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)
            if entity["title"] not in title2id:
                missing_pages += 1
            else:
                kb2id[entity["entity_id"]] = title2id[entity["title"]]
            n += 1
    if logger:
        logger.info("missing {}/{} pages".format(missing_pages, n))
    return kb2id


def __load_test(test_filename, kb2id, wikipedia_id2local_id, logger):
    test_samples = []
    with open(test_filename, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            record = json.loads(line)
            record["label"] = str(record["label_id"])

            # for tac kbp we should use a separate knowledge source to get the entity id (label_id)
            if kb2id and len(kb2id) > 0:
                if record["label"] in kb2id:
                    record["label_id"] = kb2id[record["label"]]
                else:
                    continue

            # check that each entity id (label_id) is in the entity collection
            elif wikipedia_id2local_id and len(wikipedia_id2local_id) > 0:
                try:
                    key = int(record["label"].strip())
                    if key in wikipedia_id2local_id:
                        record["label_id"] = wikipedia_id2local_id[key]
                    else:
                        continue
                except:
                    continue

            # LOWERCASE EVERYTHING !
            record["context_left"] = record["context_left"].lower()
            record["context_right"] = record["context_right"].lower()
            record["mention"] = record["mention"].lower()
            test_samples.append(record)

    if logger:
        logger.info("{}/{} samples considered".format(len(test_samples), len(lines)))
    return test_samples


def _get_test_samples(
    test_filename, test_entities_path, title2id, wikipedia_id2local_id, logger
):
    kb2id = None
    if test_entities_path:
        kb2id = __map_test_entities(test_entities_path, title2id, logger)
    test_samples = __load_test(test_filename, kb2id, wikipedia_id2local_id, logger)
    return test_samples


def _process_biencoder_dataloader(samples, tokenizer, biencoder_params):
    _, tensor_data = process_mention_data(
        samples,
        tokenizer,
        biencoder_params["max_context_length"],
        biencoder_params["max_cand_length"],
        silent=True,
        logger=None,
        debug=biencoder_params["debug"],
    )
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=biencoder_params["eval_batch_size"]
    )
    return dataloader


def _run_biencoder(biencoder, dataloader, candidate_encoding, top_k=100, indexer=None):
    biencoder.model.eval()
    labels = []
    nns = []
    all_scores = []
    for batch in tqdm(dataloader):
        context_input, _, label_ids = batch
        with torch.no_grad():
            if indexer is not None:
                context_encoding = biencoder.encode_context(context_input).numpy()
                context_encoding = np.ascontiguousarray(context_encoding)
                scores, indicies = indexer.search_knn(context_encoding, top_k)
            else:
                scores = biencoder.score_candidate(
                    context_input, None, cand_encs=candidate_encoding  # .to(device)
                )
                scores, indicies = scores.topk(top_k)
                scores = scores.data.numpy()
                indicies = indicies.data.numpy()

        labels.extend(label_ids.data.numpy())
        nns.extend(indicies)
        all_scores.extend(scores)
    return labels, nns, all_scores


def _process_crossencoder_dataloader(context_input, label_input, crossencoder_params):
    tensor_data = TensorDataset(context_input, label_input)
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=crossencoder_params["eval_batch_size"]
    )
    return dataloader


def _run_crossencoder(crossencoder, dataloader, logger, context_len, device="cuda"):
    crossencoder.model.eval()
    accuracy = 0.0
    crossencoder.to(device)

    res = evaluate(crossencoder, dataloader, device, logger, context_len, silent=False)
    accuracy = res["normalized_accuracy"]
    logits = res["logits"]

    predictions = np.argsort(logits, axis=1)
    return accuracy, predictions, logits


def load_models(args, logger=None):
    # load biencoder model
    if logger:
        logger.info("loading biencoder model")
    with open(args.biencoder_config) as json_file:
        biencoder_params = json.load(json_file)
        biencoder_params["path_to_model"] = args.biencoder_model
    biencoder = load_biencoder(biencoder_params)

    crossencoder = None
    crossencoder_params = None
    if not args.fast:
        # load crossencoder model
        if logger:
            logger.info("loading crossencoder model")
        with open(args.crossencoder_config) as json_file:
            crossencoder_params = json.load(json_file)
            crossencoder_params["path_to_model"] = args.crossencoder_model
        crossencoder = load_crossencoder(crossencoder_params)

    # load candidate entities
    if logger:
        logger.info("loading candidate entities")
    (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        faiss_indexer,
    ) = _load_candidates(
        args.entity_catalogue,
        args.entity_encoding,
        faiss_index=args.faiss_index,
        index_path=args.index_path,
        logger=logger,
    )

    return (
        biencoder,
        biencoder_params,
        crossencoder,
        crossencoder_params,
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        faiss_indexer,
    )


class EntityLinker:
    def __init__(self, args, logger=None) -> None:
        self.args = args
        self.logger = logger
        # Load NER model
        self.ner_model = NER.get_model()
        # load biencoder model
        if logger:
            logger.info("loading biencoder model")
        with open(args.biencoder_config) as json_file:
            biencoder_params = json.load(json_file)
            biencoder_params["path_to_model"] = args.biencoder_model
        biencoder = load_biencoder(biencoder_params)

        crossencoder = None
        crossencoder_params = None
        if not args.fast:
            # load crossencoder model
            if logger:
                logger.info("loading crossencoder model")
            with open(args.crossencoder_config) as json_file:
                crossencoder_params = json.load(json_file)
                crossencoder_params["path_to_model"] = args.crossencoder_model
            crossencoder = load_crossencoder(crossencoder_params)

        # load candidate entities
        if logger:
            logger.info("loading candidate entities")
        (
            candidate_encoding,
            title2id,
            id2title,
            id2text,
            wikipedia_id2local_id,
            faiss_indexer,
        ) = _load_candidates(
            args.entity_catalogue,
            args.entity_encoding,
            faiss_index=args.faiss_index,
            index_path=args.index_path,
            logger=logger,
        )

        self.biencoder = biencoder
        self.biencoder_params = biencoder_params
        self.crossencoder = crossencoder
        self.crossencoder_params = crossencoder_params
        self.candidate_encoding = candidate_encoding
        self.title2id = title2id
        self.id2title = id2title
        self.id2text = id2text
        self.wikipedia_id2local_id = wikipedia_id2local_id
        self.faiss_indexer = faiss_indexer
        self.id2url = {
            v: "https://en.wikipedia.org/wiki?curid=%s" % k
            for k, v in wikipedia_id2local_id.items()
        }

    def link_text(self, text):
        # Identify mentions
        samples = _annotate(self.ner_model, [text])

        _print_colorful_text(text, samples)

        # don't look at labels
        keep_all = True

        # prepare the data for biencoder
        logger.info("preparing data for biencoder")
        dataloader = _process_biencoder_dataloader(
            samples, self.biencoder.tokenizer, self.biencoder_params
        )

        # run biencoder
        logger.info("run biencoder")
        top_k = args.top_k
        labels, nns, scores = _run_biencoder(
            self.biencoder,
            dataloader,
            self.candidate_encoding,
            top_k,
            self.faiss_indexer,
        )

        # print biencoder prediction
        idx = 0
        linked_entities = []
        for entity_list, sample in zip(nns, samples):
            e_id = entity_list[0]
            e_title = self.id2title[e_id]
            e_text = self.id2text[e_id]
            e_url = self.id2url[e_id]
            linked_entities.append(
                {
                    "idx": idx,
                    "sample": sample,
                    "entity_id": e_id,
                    "entity_title": e_title,
                    "entity_text": e_text,
                    "url": e_url,
                    "crossencoder": False,
                }
            )
            idx += 1

        if args.fast:
            # use only biencoder
            return {"samples": samples, "linked_entities": linked_entities}

        # prepare crossencoder data
        context_input, candidate_input, label_input = prepare_crossencoder_data(
            self.crossencoder.tokenizer,
            samples,
            labels,
            nns,
            self.id2title,
            self.id2text,
            keep_all,
        )

        context_input = modify(
            context_input, candidate_input, self.crossencoder_params["max_seq_length"]
        )

        dataloader = _process_crossencoder_dataloader(
            context_input, label_input, self.crossencoder_params
        )

        # run crossencoder and get accuracy
        accuracy, index_array, unsorted_scores = _run_crossencoder(
            self.crossencoder,
            dataloader,
            logger,
            context_len=self.biencoder_params["max_context_length"],
        )

        # print crossencoder prediction
        idx = 0
        linked_entities = []
        for entity_list, index_list, sample in zip(nns, index_array, samples):
            e_id = entity_list[index_list[-1]]
            e_title = self.id2title[e_id]
            e_text = self.id2text[e_id]
            e_url = self.id2url[e_id]
            _print_colorful_prediction(
                idx, sample, e_id, e_title, e_text, e_url, args.show_url
            )
            linked_entities.append(
                {
                    "idx": idx,
                    "sample": sample,
                    "entity_id": e_id,
                    "entity_title": e_title,
                    "entity_text": e_text,
                    "url": e_url,
                    "crossencoder": True,
                }
            )
            idx += 1
        return {"samples": samples, "linked_entities": linked_entities}

    def run(self):
        self.logger.info("interactive mode")
        while True:
            # Interactive
            text = input("insert text:")
            output = self.link_text(text)
            samples = output["samples"]
            linked_entities = output["linked_entities"]
            _print_colorful_text(text, samples)
            for e in linked_entities:
                _print_colorful_prediction(
                    e["idx"],
                    e["sample"],
                    e["entity_id"],
                    e["entity_title"],
                    e["entity_text"],
                    e["url"],
                    self.args.show_url,
                )
            print()
    
    def create_app(self):
        app = FastAPI()
        @app.get("/api/")
        async def entity_link():
            pass
        
        return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # biencoder
    parser.add_argument(
        "--biencoder_model",
        dest="biencoder_model",
        type=str,
        default="models/biencoder_wiki_large.bin",
        help="Path to the biencoder model.",
    )
    parser.add_argument(
        "--biencoder_config",
        dest="biencoder_config",
        type=str,
        default="models/biencoder_wiki_large.json",
        help="Path to the biencoder configuration.",
    )
    parser.add_argument(
        "--entity_catalogue",
        dest="entity_catalogue",
        type=str,
        # default="models/tac_entity.jsonl",  # TAC-KBP
        default="models/entity.jsonl",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )
    parser.add_argument(
        "--entity_encoding",
        dest="entity_encoding",
        type=str,
        # default="models/tac_candidate_encode_large.t7",  # TAC-KBP
        default="models/all_entities_large.t7",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )

    # crossencoder
    parser.add_argument(
        "--crossencoder_model",
        dest="crossencoder_model",
        type=str,
        default="models/crossencoder_wiki_large.bin",
        help="Path to the crossencoder model.",
    )
    parser.add_argument(
        "--crossencoder_config",
        dest="crossencoder_config",
        type=str,
        default="models/crossencoder_wiki_large.json",
        help="Path to the crossencoder configuration.",
    )

    parser.add_argument(
        "--top_k",
        dest="top_k",
        type=int,
        default=10,
        help="Number of candidates retrieved by biencoder.",
    )

    # output folder
    parser.add_argument(
        "--output_path",
        dest="output_path",
        type=str,
        default="output",
        help="Path to the output.",
    )

    parser.add_argument(
        "--fast", dest="fast", action="store_true", help="only biencoder mode"
    )

    parser.add_argument(
        "--show_url",
        dest="show_url",
        action="store_true",
        help="whether to show entity url in interactive mode",
    )

    parser.add_argument(
        "--faiss_index",
        type=str,
        default=None,
        help="whether to use faiss index",
    )

    parser.add_argument(
        "--index_path",
        type=str,
        default=None,
        help="path to load indexer",
    )

    args = parser.parse_args()

    logger = utils.get_logger(args.output_path)

    EntityLinker(args, logger).run()
