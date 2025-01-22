from ultralytics.models import yolo
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from ultralytics.utils.checks import check_imgsz
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.utils.ops import Profile
import torch
import json
from .utils import crop_images
import numpy as np


class SliceDetectionValidator(yolo.detect.DetectionValidator):

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Supports validation of a pre-trained model if passed or a model being trained if trainer is passed (trainer
        gets priority).
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            self.args.half = self.device.type != "cpu"  # force FP16 val during training
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            # self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            # self.model = model
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            slice_size = 1280
            imgsz = check_imgsz(slice_size, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = 1  # export.py models default to batch-size 1
                LOGGER.info(f"Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val

        # #################### debug ####################
        import psutil
        import os

        process = psutil.Process(os.getpid())
        def print_mem_info(process, message="mem"):
            mem_info = process.memory_info()
            print("")
            print(message)
            print(f"RSS: {mem_info.rss / (1024 ** 2):.2f} MB")  # 以MB为单位显示
            print(f"VMS: {mem_info.vms / (1024 ** 2):.2f} MB")  # 以MB为单位显示
        print_mem_info(process)
        # #################### end debug ####################
        with torch.no_grad():
            for batch_i, batch in enumerate(bar):
                print_mem_info(process, "batch_i start:")
                self.run_callbacks("on_val_batch_start")
                self.batch_i = batch_i
                # print(f"{batch['im_file']}: {batch['img'].shape}")
                # Preprocess
                with dt[0]:
                    print_mem_info(process, "before preprocess:")
                    batch = self.preprocess(batch, device="cpu")
                    print_mem_info(process, "after preprocess:")

                # Inference
                with dt[1]:
                    # if batch_i == 0:
                    #     LOGGER.info(f"slice num: {len(img_slices)}, slice size: {imgsz}")

                    preds_all = []
                    print_mem_info(process, "before inference:")
                    for offset_i, img_i in self.crop_image_batch(batch, imgsz, step=0.9, batch_size=32):
                        preds_i = model(img_i, augment=augment)[0]
                        preds_i[:, :2, :] += offset_i
                        preds_all.append(torch.cat(tuple(preds_i), dim=-1))
                    preds = torch.cat(preds_all, dim=-1)
                    preds = preds.unsqueeze(0)
                    # preds = model(batch["img"], augment=augment)
                    print_mem_info(process, "after inference:")
                # Loss
                with dt[2]:
                    if self.training:
                        raise RuntimeError("slice training method is not supported")

                # Postprocess
                with dt[3]:
                    preds = self.postprocess(preds)
                    print_mem_info(process, "after postprocess:")

                self.update_metrics(preds, batch)
                if self.args.plots and batch_i < 0:
                    self.plot_val_samples(batch, batch_i)
                    self.plot_predictions(batch, preds, batch_i)

                self.run_callbacks("on_val_batch_end")
                print_mem_info(process, "after metrics:")
                del batch
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image"
                % tuple(self.speed.values())
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def preprocess(self, batch, device=None):
        """Preprocesses batch of images for YOLO training."""
        device = self.device if device is None else device
        batch["img"] = batch["img"].to(device, non_blocking=True)
        # batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255

        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = (
                [
                    torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                    for i in range(nb)
                ]
                if self.args.save_hybrid
                else []
            )  # for autolabelling

        return batch

    def crop_image_batch(self, batch, imgsz, step=0.9, batch_size=4):
        offsets, img_slices = crop_images(batch['img'], (imgsz, imgsz), step)
        start = 0
        while True:
            end = start + batch_size
            # shape: (b, 2, 1)
            offsets_batch = np.array(offsets[start:end])[..., np.newaxis]
            offsets_batch = torch.from_numpy(offsets_batch).to(self.device)
            img_batch = img_slices[start:end]
            img_batch = torch.concat(img_batch, axis=0)
            img_batch = img_batch.to(self.device)
            img_batch = (img_batch.half() if self.args.half else img_batch.float()) / 255
            yield offsets_batch, img_batch.to(self.device)

            if end >= len(offsets):
                break
            start = end 


