## J
- [ ] install the openhands env inside our docker
- [ ] move necessary files to outside?
- [ ] stream the logging
- [ ] cost estimator
- [ ] remove the need to remove and restart the docker
- [ ] continue previous experiments w/ checkpoint
- [ ] human interruption
- [ ] OAI Deep research will ask for clarification questions; stop if neccessary credential is not provided
- [ ] need an good abstraction for add task; all related config in one file!!!
- [ ] clean up my docker file
- [ ] clean up the readme
- [ ] clean up benchmark
- [ ] abstract prompt and other config for user to easily customize their use case; too many if else in main_loop.
- [ ] main loop should be the main.py, which contains all critical parts.
- [ ] need to automatically set up openhands config (starter file dir, etc.) --> may need to update benchmark workspace directory?
- [ ] remove hardcode dir. (e.g. main_loop.py)
- [ ] rename src to curie ? 
- [ ] clean up all names w/ langgraph-exp-agent
- [ ] add unit test
- [ ] put helper funciton to utils
- [ ] automate openhands credential setup.
- [ ] fix the openhands version
- [ ] break down long system prompt
- [ ] all manually runned docker commands can use 'import docker'
- [ ] all benchmark questions should start with the directory under starter_file
- [ ] we can fine-tune the coding difficulty level of each task by cutting the instructions to test openhands abilities.


## P
- [ ] cleanup agents files: we can use a common abstraction
- [ ] support for other LLM models other than GPT
- [ ] support for other agents 
- [ ] implement search space agent: possible candidates [1](https://github.com/google-deepmind/long-form-factuality/tree/main/eval/safe) [1a](https://cobusgreyling.medium.com/agentic-search-augmented-factuality-evaluator-safe-for-llms-9b1ff7aeb784) [1b](https://github.com/google-deepmind/long-form-factuality/blob/main/eval/safe/query_serper.py)
- [ ] Draw a precise internal impl architecture?
- [ ] After system runs well, draw system diagram figures for Sec 3. 

paper appendix
- [ ] details about starter files in the benchmark