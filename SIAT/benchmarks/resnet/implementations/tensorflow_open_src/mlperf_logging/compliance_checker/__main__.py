import sys

from . import mlp_compliance


parser = mlp_compliance.get_parser()
args = parser.parse_args()

config_file = args.config or f'{args.ruleset}/common.yaml'

checker = mlp_compliance.make_checker(
    args.ruleset,
    args.quiet,
    args.werror,
)

valid, system_id, benchmark, result = mlp_compliance.main(args.filename, config_file, checker)

print(valid)
print(system_id)
print(benchmark)
print(result)

if not valid:
    sys.exit(1)
