$PROJECT = 'anamic'

$ACTIVITIES = ['version_bump', 'tag', 'push_tag', 'pypi', 'ghrelease']

$VERSION_BUMP_PATTERNS = [('anamic/__init__.py', '__version__\s*=.*', "__version__ = '$VERSION'"),
                          ('setup.py', 'version\s*=.*,', "version='$VERSION',")
                          ]

$PUSH_TAG_REMOTE = 'git@github.com:brouhardlab/anamic.git'
$GITHUB_ORG = 'brouhardlab'
$GITHUB_REPO = 'anamic'
