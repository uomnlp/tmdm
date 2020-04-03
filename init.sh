#!/usr/bin/env bash
#
# ~ Initialises the repo...
#
#

while [[ -z "$PROJECT_NAME" || "$PROJECT_NAME" =~ [^a-zA-Z] ]]; do
  read -r -p "Project Name: " PROJECT_NAME
  if [[ -z "$PROJECT_NAME" || "$PROJECT_NAME" =~ [^a-zA-Z] ]]; then
    echo "Letters only!"
  fi
done

NAME_LOWER="${PROJECT_NAME,,}"
read -r -p "Author: " AUTHOR
read -r -p "Author email: " EMAIL
read -r -p "Version (leave blank for 0.0.1): " VERSION
if [[ -z "$VERSION" ]]; then
  VERSION="0.0.1"
fi
SHORT_VERSION="${VERSION%.*}"
read -r -p "Description: " DESCRIPTION
read -r -p "URL: " URL
read -r -p "License (Leave blank for GPLv3): " LICENSE

if [[ ! -z "$VERSION" ]]; then
  LICENSE="GPLv3"
fi
echo "Moving project folder from 'project' to $NAME_LOWER"
mv project "$NAME_LOWER"
echo "Replacing in setup.py..."
echo "'PROJECT_NAME' for $PROJECT_NAME"
echo "'AUTHOR' for $AUTHOR"
echo "'EMAIL' for $EMAIL"
echo "'VERSION' for $VERSION"
echo "'DESCRIPTION' for $DESCRIPTION"
echo "'URL' for $URL"
echo "'LICENSE' for $LICENSE"
sed "s/PROJECT_NAME/$PROJECT_NAME/g;s/AUTHOR/$AUTHOR/g;s/EMAIL/$EMAIL/g;s/VERSION/$VERSION/g;s/DESCRIPTION/$DESCRIPTION/g;s|URL|$URL|g;s/LICENSE/$LICENSE/g;" setup.py >done-setup.py
echo "Done!"
echo "in docs/conf.py..."
echo "'PROJECT_NAME' for $PROJECT_NAME"
echo "'COPYRIGHT' for \"$(date +'%Y'), $AUTHOR\""
COPYRIGHT="$(date +'%Y'), $AUTHOR"
echo "'AUTHOR' for $AUTHOR"
echo "'SHORT_VERSION' for $SHORT_VERSION"
echo "'RELEASE' for $VERSION"
sed "s/PROJECT_NAME/$PROJECT_NAME/g;s/AUTHOR/$AUTHOR/g;s/COPYRIGHT/$COPYRIGHT/g;s/RELEASE/$VERSION/g;s/SHORT_VERSION/$SHORT_VERSION/g;" docs/conf.py >docs/done-conf.py
