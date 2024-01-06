FURTHER_ARGS :=

# This command require ONE and only ONE argument.
CMDS_REQUIRING_ONE_ARG := demo
ifneq ($(filter $(firstword $(MAKECMDGOALS)),$(CMDS_REQUIRING_ONE_ARG)),)
  FURTHER_ARGS := $(filter-out $(firstword $(MAKECMDGOALS)), $(MAKECMDGOALS))
  # The containers are not targets, let's make them silently fail
  $(eval $(FURTHER_ARGS):;@true)
  $(eval .PHONY: $(FURTHER_ARGS))
  ifneq ($(words $(FURTHER_ARGS)),1)
  	$(error This command requires exactly one argument to be specified. Please check the help for more information.)
  endif
endif

# Add here commands that support additional arguments
CMDS_WITH_ARGS := test
ifneq ($(filter $(firstword $(MAKECMDGOALS)),$(CMDS_WITH_ARGS)),)
  FURTHER_ARGS := $(wordlist 2,999,$(MAKECMDGOALS))
  # The args are not targets, let's make them silently fail
  $(eval $(subst :,\:,$(subst %,\%, $(FURTHER_ARGS))):;@true)
  $(eval .PHONY: $(subst :,\:, $(subst ;,\;, $(FURTHER_ARGS))))
endif

# define standard colors
ifneq (,$(findstring xterm,${TERM}))
	BLACK        := $(shell tput -Txterm setaf 0)
	RED          := $(shell tput -Txterm setaf 1)
	GREEN        := $(shell tput -Txterm setaf 2)
	YELLOW       := $(shell tput -Txterm setaf 3)
	LIGHTPURPLE  := $(shell tput -Txterm setaf 4)
	PURPLE       := $(shell tput -Txterm setaf 5)
	BLUE         := $(shell tput -Txterm setaf 6)
	WHITE        := $(shell tput -Txterm setaf 7)
	RESET := $(shell tput -Txterm sgr0)
else
	BLACK        := ""
	RED          := ""
	GREEN        := ""
	YELLOW       := ""
	LIGHTPURPLE  := ""
	PURPLE       := ""
	BLUE         := ""
	WHITE        := ""
	RESET        := ""
endif

# set target color
NOTIFICATION_COLOR := $(BLUE)

.PHONY: all help
all: help
help:
	@echo "Here is a list of make commands with the corresponding description"
	@echo
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(NOTIFICATION_COLOR)%-30s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: test demo coverage
test: ## Run tests
	@PYTHONPATH=. pytest -p no:warnings -s -vvv ${FURTHER_ARGS}
demo: ## Run demo
	@python ${FURTHER_ARGS}
coverage: ## Run tests with coverage
	@coverage run -m pytest && coverage combine && coverage html && coverage report