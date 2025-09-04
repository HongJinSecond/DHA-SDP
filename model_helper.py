from utils.util import parse_jit_args,build_model_tokenizer_config
from peft import LoraConfig,get_peft_model
from models.ConcatModel import ConcatModel
from models.SingleModel import SingleModel
from models.ManualModel import ManualModel
from config import *
import torch



if __name__ == "__main__":
    args = parse_jit_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for base_model in ["concat"]:
        for pretrained in ["codebert"]:
            print(f"——————————————————run base model {base_model} on encoder {pretrained}————————————————————")
            args.base_model=base_model
            args.pretrained_model=pretrained

            model, tokenizer, config = build_model_tokenizer_config(args)
            if args.pretrained_model in ["codet5"]:
                peft_config = LoraConfig(inference_mode=False, r=64, lora_alpha=32,
                                         lora_dropout=0.1, target_modules=["q", "v"])
            elif args.pretrained_model in ["plbart"]:
                peft_config = LoraConfig(inference_mode=False, r=64, lora_alpha=32,
                                         lora_dropout=0.1, target_modules=["q_proj", "v_proj"])
            elif args.pretrained_model in ["codebert", "graphcodebert", "unixcoder"]:
                peft_config = LoraConfig(inference_mode=False, r=64, lora_alpha=32,
                                         lora_dropout=0.1, target_modules=["query", "value"])
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

            if args.base_model == "concat":
                mymodel = ConcatModel(model, config, tokenizer, args)
            elif args.base_model == "single":
                mymodel = SingleModel(model, config, tokenizer, args)
            elif args.base_model == "manual":
                mymodel = ManualModel(model, config, tokenizer, args)
            else:
                raise ValueError(f"Invalid base model: {args.base_model}")
            model_name = f"{args.base_model}-{args.pretrained_model}-final.pt"
            # Merge the overall lora into baseline models.
            save_dir = os.path.join(args.output_dir, f"checkpoints/{args.base_model}")
            mymodel.encoder.merge_and_unload()
            # Save the state dictionary.
            mymodel.load_state_dict(torch.load(os.path.join(save_dir, model_name),weights_only=True),strict=True)
            mymodel.encoder = mymodel.encoder.base_model.model
            torch.save(mymodel.state_dict(), os.path.join(save_dir, model_name))