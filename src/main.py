from utils import argmanager


def cli():
	args = argmanager.fetch_main_parser()
	print(args)
	# Call commands based on subcommand


if __name__ == "__main__":
	cli()
