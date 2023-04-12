#!/bin/bash
# download dependencies
/bin/bash dependencies/get_dependencies.sh

# move dependencies to docker/dependencies
mv dependencies/*.tar.gz docker/dependecies

# download intel OneAPI 
/bin/bash optional/get_oneapi.sh

# move to docker/optional
mv optional/*.sh docker/optional/
