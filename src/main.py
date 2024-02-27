from utils import argmanager
import variant_scoring
import variant_scoring_per_chrom
import variant_scoring_per_chunk
import variant_shap
import variant_summary_across_folds
import variant_annotation


def cli():
	args = argmanager.fetch_main_parser()
	if args.subcommand == "score":
		variant_scoring.main(args)
	elif args.subcommand == "score-per-chrom":
		variant_scoring_per_chrom.main(args)
	elif args.subcommand == "score-per-chunk":
		variant_scoring_per_chunk.main(args)
	elif args.subcommand == "shap":
		variant_shap.main(args)
	elif args.subcommand == "summary":
		variant_summary_across_folds.main(args)
	elif args.subcommand == "annotate":
		variant_annotation.main(args)


if __name__ == "__main__":
	cli()
