import argparse
import subprocess


def run_script(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)


def main():
    parser = argparse.ArgumentParser(description="AMNLT Experiment Launcher")
    parser.add_argument("--aproach", type=str, required=True,
                        choices=["divide-conquer", "base-holistic", "unfolding", "language-modeling"],
                        help="Approach to run")
    parser.add_argument("--model", type=str, required=True,
                        choices=["crnn", "fcn", "cnnt2d", "van", "origaminet", "smt", "dan", "trocr", "paligemma"],
                        help="Model to run")
    parser.add_argument("--modality", type=str, required=True,
                        choices=["music", "lyrics"],
                         help="Modality of the input data (only for divide-conquer)")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "test"],
                        help="Mode to run: train or test")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["einsiedeln", "salzinnes", "solesmes", "gregosynth"],
                        help="Dataset to use")

    args = parser.parse_args()

    
    valid_combinations = {
        "divide-conquer": {"crnn"},
        "base-holistic": {"crnn"},
        "unfolding": {"fcn", "crnn", "cnnt2d", "van", "origaminet"},
        "language-modeling": {"smt", "dan", "trocr", "paligemma"}
    }

    # Validate model-approach combination
    if args.model not in valid_combinations[args.aproach]:
        raise ValueError(f"The model '{args.model}' is not valid for the approach '{args.aproach}'. "
                        f"Valid options: {', '.join(valid_combinations[args.aproach])}")

    # Validate modality only for divide-conquer
    if args.aproach == "divide-conquer":
        if not args.modality:
            raise ValueError("The '--modality' argument is required for the 'divide-conquer' approach.")
    else:
        if args.modality:
            raise ValueError(f"The '--modality' argument is not applicable to the '{args.aproach}' approach.")
        
    key = (args.aproach, args.model, args.modality, args.mode) if args.aproach == "divide-conquer" \
    else (args.aproach, args.model, args.mode)

    command_template = entrypoints.get(key)

    if not command_template:
        raise ValueError(f"No command found for the combination: {key}")

    command = command_template.format(
        dataset=args.dataset,
        modality=args.modality
    )

    entrypoints = {
        # Divide & Conquer          
        ("divide-conquer", "train"):
            "train.py --ds_name {dataset}_{modality} --model_name crnn --encoding_type new_gabc --ctc greedy",
        ("divide-conquer", "test"):
            "test.py --ds_name {dataset}_{modality} --model_name crnn --encoding_type new_gabc --ctc greedy --checkpoint_path weights/{dataset}_{modality}/{model}.ckpt",

        # Base Holistic
        ("base-holistic", "train"): 
            "train.py --ds_name {dataset} --model_name crnn --encoding_type new_gabc --ctc greedy",
        ("base-holistic", "test"):
            "test.py --ds_name {dataset} --model_name crnn --encoding_type new_gabc --ctc greedy --checkpoint_path weights/{dataset}/{model}.ckpt",

        # Unfolding
        ("unfolding", "fcn", "train"): 
            "train.py --ds_name {dataset} --model_name fcn --encoding_type new_gabc --ctc greedy --batch_size 1",
        ("unfolding", "fcn", "test"):
            "test.py --ds_name {dataset} --model_name fcn --encoding_type new_gabc --ctc greedy --checkpoint_path weights/{dataset}/fcn.ckpt",
            
        ("unfolding", "crnn", "train"): 
            "train.py --ds_name {dataset} --model_name crnnunfolding --encoding_type new_gabc --ctc greedy --batch_size 1",
        ("unfolding", "crnn", "test"):
            "test.py --ds_name {dataset} --model_name crnnunfolding --encoding_type new_gabc --ctc greedy --checkpoint_path weights/{dataset}/crnnunfolding.ckpt",
            
        ("unfolding", "cnnt2d", "train"): 
            "train.py --ds_name {dataset} --model_name cnnt2d --encoding_type new_gabc --ctc greedy --batch_size 1",
        ("unfolding", "cnnt2d", "test"):
            "test.py --ds_name {dataset} --model_name cnnt2d --encoding_type new_gabc --ctc greedy --checkpoint_path weights/{dataset}/cnnt2d.ckpt",
            
        ("unfolding", "van", "train"): 
            "train.py --ds_name {dataset} --model_name van --encoding_type new_gabc --ctc greedy --batch_size 1",
        ("unfolding", "van", "test"):
            "test.py --ds_name {dataset} --model_name van --encoding_type new_gabc --ctc greedy --checkpoint_path weights/{dataset}/van.ckpt",
            
        ("unfolding", "origaminet", "train"): 
            "train.py --gin configs/{dataset}.gin",
        ("unfolding", "origaminet", "test"):
            "test.py --gin configs/{dataset}.gin --checkpoint weights/{dataset}/origaminet.ckpt",

        # Language Modeling
        ("language-modeling", "smt", "train"): 
            "train_smt.py --config_path config/{dataset}/{dataset}.json --dataset_name {dataset}",
        ("language-modeling", "smt", "test"):
            "test_smt.py --config_path config/{dataset}/{dataset}.json --dataset_name {dataset} --checkpoint_path weights/{dataset}/smt.ckpt",
            
        ("language-modeling", "dan", "train"): 
            "train_dan.py --config_path config/{dataset}/{dataset}.json --dataset_name {dataset}",
        ("language-modeling", "dan", "test"):
            "test_dan.py --config_path config/{dataset}/{dataset}.json --dataset_name {dataset} --checkpoint_path weights/{dataset}/dan.ckpt",
            
        ("language-modeling", "trocr", "train"): 
            "train_trocr.py --ds_name {dataset}",
        ("language-modeling", "trocr", "test"):
            "test_trocr.py --ds_name {dataset} --checkpoint_path weights/{dataset}/trocr.ckpt",
            
        ("language-modeling", "paligemma", "train"): 
            "train_paligemma.py --ds_name {dataset}",
        ("language-modeling", "paligemma", "test"):
            "test_paligemma.py --ds_name {dataset} --checkpoint_path weights/{dataset}/paligemma.ckpt",
    }


    script_path = entrypoints.get((args.model, args.mode))
    if not script_path:
        raise ValueError(f"No script defined for model '{args.model}' in mode '{args.mode}'")

    run_script(f"python -u {script_path}")


if __name__ == "__main__":
    main()
