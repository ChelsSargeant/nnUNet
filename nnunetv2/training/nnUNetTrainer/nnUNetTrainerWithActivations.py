import os
import torch
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerWithActivations(nnUNetTrainer):
    """
    nnU-Net trainer that saves intermediate activations via forward hooks.

    IMPORTANT: nnUNetTrainer.__init__ introspects the subclass signature.
    Therefore, we must NOT add *args/**kwargs or extra parameters here.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device,
    ):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )

        # Feature flags (safe to set AFTER super().__init__)
        self.save_activations = True

        # Default: save every epoch. Override via environment variable if desired.
        # Example (PowerShell): $env:NNUNET_ACT_SAVE_FREQ="5"
        freq = os.environ.get("NNUNET_ACT_SAVE_FREQ", "1")
        try:
            self.activation_save_frequency = max(1, int(freq))
        except ValueError:
            self.activation_save_frequency = 1

        self._activation_storage = {}
        self._hooks = []

    def initialize(self):
        super().initialize()
        if self.save_activations:
            self._register_activation_hooks()

    def _register_activation_hooks(self):
        self.print_to_log_file("Registering activation hooks")

        def hook_fn(name):
            def hook(module, inp, out):
                # store only tensors; detach & move to CPU immediately
                if torch.is_tensor(out):
                    self._activation_storage[name] = out.detach().cpu()
                elif isinstance(out, (list, tuple)):
                    tensors = [o.detach().cpu() for o in out if torch.is_tensor(o)]
                    if tensors:
                        self._activation_storage[name] = tensors
            return hook

        # Try common nnU-Net v2 attributes first; fall back gracefully if architecture differs
        registered = 0

        if hasattr(self.network, "encoder") and hasattr(self.network.encoder, "stages"):
            for i, stage in enumerate(self.network.encoder.stages):
                self._hooks.append(stage.register_forward_hook(hook_fn(f"encoder_stage_{i}")))
                registered += 1

        if hasattr(self.network, "encoder") and hasattr(self.network.encoder, "bottleneck"):
            self._hooks.append(self.network.encoder.bottleneck.register_forward_hook(hook_fn("bottleneck")))
            registered += 1

        # Fallback: if the above didn’t register anything, hook selected modules by type/name
        if registered == 0:
            self.print_to_log_file("Warning: encoder/bottleneck not found; using fallback module hooks")
            for name, module in self.network.named_modules():
                # conservative choice: hook only convolution blocks to avoid huge overhead
                if isinstance(module, (torch.nn.Conv3d, torch.nn.Conv2d)):
                    if "seg" in name.lower():  # skip final segmentation conv if you want
                        continue
                    self._hooks.append(module.register_forward_hook(hook_fn(name)))
                    registered += 1
                    if registered >= 16:  # cap hooks to keep I/O manageable
                        break

        self.print_to_log_file(f"Activation hooks registered: {registered}")

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)

        if not self.save_activations:
            return

        if (self.current_epoch % self.activation_save_frequency) != 0:
            self._activation_storage.clear()
            return

        self._save_activations_to_disk()

    def _save_activations_to_disk(self):
        base_folder = join(self.output_folder, "activations")
        epoch_folder = join(base_folder, f"epoch_{self.current_epoch}")
        maybe_mkdir_p(epoch_folder)

        for name, act in self._activation_storage.items():
            safe_name = name.replace("/", "_").replace("\\", "_").replace(":", "_")
            if isinstance(act, list):
                for i, a in enumerate(act):
                    np.save(join(epoch_folder, f"{safe_name}_{i}.npy"), a.numpy())
            else:
                np.save(join(epoch_folder, f"{safe_name}.npy"), act.numpy())

        self.print_to_log_file(
            f"Saved activations for epoch {self.current_epoch} "
            f"({len(self._activation_storage)} entries) to {epoch_folder}"
        )

        # Avoid memory growth across epochs
        self._activation_storage.clear()

        # # Remove hooks to avoid memory leaks
        # for h in self._hooks:
        #     h.remove()
        # self._hooks.clear()