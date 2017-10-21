# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""gcloud CLI tree generators for non-gcloud CLIs.

A CLI tree is generated by using the root command plus `help` or `--help`
arguments to do a DFS traversal. Each node is generated from a man-ish style
runtime document.

Supported CLI commands have their own runtime help document quirks, so each is
handled by an ad-hoc parser. The parsers rely on consistency within commands
and between command releases.
"""

import abc
import json
import os
import re
import subprocess

from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import files


class NoCliTreeGeneratorForCommand(exceptions.Error):
  """Command does not have a CLI tree generator."""


def _Flag(name, description='', value=None, default=None, type_='string',
          is_global=False, is_required=False):
  """Initializes and returns a flag dict node."""
  return {
      cli_tree.LOOKUP_ATTR: {},
      cli_tree.LOOKUP_CATEGORY: '',
      cli_tree.LOOKUP_DEFAULT: default,
      cli_tree.LOOKUP_DESCRIPTION: description,
      cli_tree.LOOKUP_GROUP: '',
      cli_tree.LOOKUP_IS_GLOBAL: is_global,
      cli_tree.LOOKUP_IS_HIDDEN: False,
      cli_tree.LOOKUP_IS_REQUIRED: is_required,
      cli_tree.LOOKUP_NAME: name,
      cli_tree.LOOKUP_VALUE: value,
      cli_tree.LOOKUP_TYPE: type_,
  }


def _Positional(name, description='', default=None, nargs='0'):
  """Initializes and returns a positional dict node."""
  return {
      cli_tree.LOOKUP_DEFAULT: default,
      cli_tree.LOOKUP_DESCRIPTION: description,
      cli_tree.LOOKUP_NAME: name,
      cli_tree.LOOKUP_NARGS: nargs,
  }


def _Command(path):
  """Initializes and returns a command/group dict node."""
  return {
      cli_tree.LOOKUP_CAPSULE: '',
      cli_tree.LOOKUP_COMMANDS: {},
      cli_tree.LOOKUP_FLAGS: {},
      cli_tree.LOOKUP_GROUPS: {},
      cli_tree.LOOKUP_IS_GROUP: False,
      cli_tree.LOOKUP_IS_HIDDEN: False,
      cli_tree.LOOKUP_PATH: path,
      cli_tree.LOOKUP_POSITIONALS: [],
      cli_tree.LOOKUP_RELEASE: 'GA',
      cli_tree.LOOKUP_SECTIONS: {},
  }


class CliTreeGenerator(object):
  """Base CLI tree generator."""

  def __init__(self, cli_name):
    self._cli_name = cli_name
    self._cli_version = None  # For memoizing GetVersion()

  @property
  def cli_name(self):
    return self._cli_name

  def Run(self, cmd):
    """Runs cmd and returns the output as a string."""
    return subprocess.check_output(cmd)

  def CliCommandExists(self):
    return files.FindExecutableOnPath(self.cli_name)

  def GetVersion(self):
    """Returns the CLI_VERSION string."""
    if not self._cli_version:
      self._cli_version = self.Run([self.cli_name, 'version']).split()[-1]
    return self._cli_version

  @abc.abstractmethod
  def GenerateTree(self):
    """Generates and returns the CLI tree dict."""
    return None


class _BqCollector(object):
  """bq help document section collector."""

  def __init__(self, text):
    self.text = text.split('\n')
    self.heading = 'DESCRIPTION'
    self.lookahead = None
    self.ignore_trailer = False

  def Collect(self, strip_headings=False):
    """Returns the heading and content lines from text."""
    content = []
    if self.lookahead:
      if not strip_headings:
        content.append(self.lookahead)
      self.lookahead = None
    heading = self.heading
    self.heading = None
    while self.text:
      line = self.text.pop(0)
      if line.startswith(' ') or not strip_headings and not self.ignore_trailer:
        content.append(line.rstrip())
    while content and not content[0]:
      content.pop(0)
    while content and not content[-1]:
      content.pop()
    self.ignore_trailer = True
    return heading, content


class BqCliTreeGenerator(CliTreeGenerator):
  """bq CLI tree generator."""

  def Run(self, cmd):
    """Runs cmd and returns the output as a string."""
    try:
      output = subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
      # bq exit code is 1 for help and --help. How do you know if help failed?
      if e.returncode != 1:
        raise
      output = e.output
    return output.replace('bq.py', 'bq')

  def AddFlags(self, command, content, is_global=False):
    """Adds flags in content lines to command."""
    while content:
      line = content.pop(0)
      name, description = line.strip().split(':', 1)
      paragraph = [description.strip()]
      default = ''
      while content and not content[0].startswith('  --'):
        line = content.pop(0).strip()
        if line.startswith('(default: '):
          default = line[10:-1]
        else:
          paragraph.append(line)
      description = ' '.join(paragraph).strip()
      if name.startswith('--[no]'):
        name = '--' + name[6:]
        type_ = 'bool'
        value = ''
      else:
        value = 'VALUE'
        type_ = 'string'
      command[cli_tree.LOOKUP_FLAGS][name] = _Flag(
          name=name,
          description=description,
          type_=type_,
          value=value,
          default=default,
          is_required=False,
          is_global=is_global,
      )

  def SubTree(self, path):
    """Generates and returns the CLI subtree rooted at path."""
    command = _Command(path)
    command[cli_tree.LOOKUP_IS_GROUP] = True
    text = self.Run([path[0], 'help'] + path[1:])

    # `bq help` lists help for all commands. Command flags are "defined"
    # by example. We don't attempt to suss that out.
    content = text.split('\n')
    while content:
      line = content.pop(0)
      if not line or not line[0].islower():
        continue
      name, text = line.split(' ', 1)
      description = [text.strip()]
      examples = []
      arguments = []
      paragraph = description
      while content and (not content[0] or not content[0][0].islower()):
        line = content.pop(0).strip()
        if line == 'Arguments:':
          paragraph = arguments
        elif line == 'Examples:':
          paragraph = examples
        else:
          paragraph.append(line)
      subcommand = _Command(path + [name])
      command[cli_tree.LOOKUP_COMMANDS][name] = subcommand
      if description:
        subcommand[cli_tree.LOOKUP_SECTIONS]['DESCRIPTION'] = '\n'.join(
            description)
      if examples:
        subcommand[cli_tree.LOOKUP_SECTIONS]['EXAMPLES'] = '\n'.join(
            examples)

    return command

  def GenerateTree(self):
    """Generates and returns the CLI tree rooted at path."""

    # Construct the tree minus the global flags.
    tree = self.SubTree([self.cli_name])

    # Add the global flags to the root.
    text = self.Run([self.cli_name, '--help'])
    collector = _BqCollector(text)
    _, content = collector.Collect(strip_headings=True)
    self.AddFlags(tree, content, is_global=True)

    # Finally add the VERSION stamp.
    tree[cli_tree.LOOKUP_CLI_VERSION] = self.GetVersion()
    tree[cli_tree.LOOKUP_VERSION] = cli_tree.VERSION

    return tree


class _GsutilCollector(object):
  """gsutil help document section collector."""

  UNKNOWN, ROOT, MAN, TOPIC = range(4)

  def __init__(self, text):
    self.text = text.split('\n')
    self.heading = 'CAPSULE'
    self.page_type = self.UNKNOWN

  def Collect(self, strip_headings=False):
    """Returns the heading and content lines from text."""
    content = []
    heading = self.heading
    self.heading = None
    while self.text:
      line = self.text.pop(0)
      if self.page_type == self.UNKNOWN:
        # The first heading distinguishes the document page type.
        if line.startswith('Usage:'):
          self.page_type = self.ROOT
          continue
        elif line == 'NAME':
          self.page_type = self.MAN
          heading = 'CAPSULE'
          continue
        elif not line.startswith(' '):
          continue
      elif self.page_type == self.ROOT:
        # The root help page.
        if line == 'Available commands:':
          heading = 'COMMANDS'
          continue
        elif line == 'Additional help topics:':
          self.heading = 'TOPICS'
          break
        elif not line.startswith(' '):
          continue
      elif self.page_type == self.MAN:
        # A command/subcommand man style page.
        if line == 'OVERVIEW':
          self.page_type = self.TOPIC
          self.heading = 'DESCRIPTION'
          break
        elif line == 'SYNOPSIS':
          self.heading = line
          break
        elif line.endswith('OPTIONS'):
          self.heading = 'FLAGS'
          break
        elif line and line[0].isupper():
          self.heading = line.split(' ', 1)[-1]
          break
      elif self.page_type == self.TOPIC:
        # A topic man style page.
        if line and line[0].isupper():
          self.heading = line
          break
      if line.startswith(' ') or not strip_headings:
        content.append(line.rstrip())
    while content and not content[0]:
      content.pop(0)
    while content and not content[-1]:
      content.pop()
    return heading, content


class GsutilCliTreeGenerator(CliTreeGenerator):
  """gsutil CLI tree generator."""

  def __init__(self, *args, **kwargs):
    super(GsutilCliTreeGenerator, self).__init__(*args, **kwargs)
    self.topics = []

  def Run(self, cmd):
    """Runs the command in cmd and returns the output as a string."""
    try:
      output = subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
      # gsutil exit code is 1 for --help depending on the context.
      if e.returncode != 1:
        raise
      output = e.output
    return output

  def AddFlags(self, command, content, is_global=False):
    """Adds flags in content lines to command."""

    def _Add(name, description):
      value = ''
      type_ = 'bool'
      default = ''
      command[cli_tree.LOOKUP_FLAGS][name] = _Flag(
          name=name,
          description=description,
          type_=type_,
          value=value,
          default=default,
          is_required=False,
          is_global=is_global,
      )

    parse = re.compile(' *((-[^ ]*,)* *(-[^ ]*) *)(.*)')
    name = None
    description = []
    for line in content:
      if line.startswith('  -'):
        if name:
          _Add(name, '\n'.join(description))
        match = parse.match(line)
        name = match.group(3)
        description = [match.group(4).rstrip()]
      elif len(line) > 16:
        description.append(line[16:].rstrip())
    if name:
      _Add(name, '\n'.join(description))

  def SubTree(self, path):
    """Generates and returns the CLI subtree rooted at path."""
    command = _Command(path)
    is_help_command = len(path) > 1 and path[1] == 'help'
    if is_help_command:
      cmd = path
    else:
      cmd = path + ['--help']
    text = self.Run(cmd)
    collector = _GsutilCollector(text)

    while True:
      heading, content = collector.Collect()
      if not heading:
        break
      elif heading == 'CAPSULE':
        if content:
          command[cli_tree.LOOKUP_CAPSULE] = content[0].split('-', 1)[1].strip()
      elif heading == 'COMMANDS':
        if is_help_command:
          continue
        for line in content:
          try:
            name = line.split()[0]
          except IndexError:
            continue
          if name == 'update':
            continue
          command[cli_tree.LOOKUP_IS_GROUP] = True
          command[cli_tree.LOOKUP_COMMANDS][name] = self.SubTree(path + [name])
      elif heading == 'FLAGS':
        self.AddFlags(command, content)
      elif heading == 'SYNOPSIS':
        commands = []
        for line in content:
          if not line:
            break
          cmd = line.split()
          if len(cmd) <= len(path):
            continue
          if cmd[:len(path)] == path:
            name = cmd[len(path)]
            if name[0].islower() and name not in ('off', 'on', 'false', 'true'):
              commands.append(name)
        if len(commands) > 1:
          command[cli_tree.LOOKUP_IS_GROUP] = True
          for name in commands:
            command[cli_tree.LOOKUP_COMMANDS][name] = self.SubTree(
                path + [name])
      elif heading == 'TOPICS':
        for line in content:
          try:
            self.topics.append(line.split()[0])
          except IndexError:
            continue
      elif heading.isupper():
        if heading.lower() == path[-1]:
          heading = 'DESCRIPTION'
        command[cli_tree.LOOKUP_SECTIONS][heading] = '\n'.join(
            [line[2:] for line in content])

    return command

  def GenerateTree(self):
    """Generates and returns the CLI tree rooted at path."""
    tree = self.SubTree([self.cli_name])

    # Add the global flags to the root.
    text = self.Run([self.cli_name, 'help', 'options'])
    collector = _GsutilCollector(text)
    while True:
      heading, content = collector.Collect()
      if not heading:
        break
      if heading == 'FLAGS':
        self.AddFlags(tree, content, is_global=True)

    # Add the help topics.
    help_command = tree[cli_tree.LOOKUP_COMMANDS]['help']
    help_command[cli_tree.LOOKUP_IS_GROUP] = True
    for topic in self.topics:
      help_command[cli_tree.LOOKUP_COMMANDS][topic] = self.SubTree(
          help_command[cli_tree.LOOKUP_PATH] + [topic])

    # Finally add the VERSION stamp.
    tree[cli_tree.LOOKUP_CLI_VERSION] = self.GetVersion()
    tree[cli_tree.LOOKUP_VERSION] = cli_tree.VERSION

    return tree


class _KubectlCollector(object):
  """Kubectl help document section collector."""

  def __init__(self, text):
    self.text = text.split('\n')
    self.heading = 'DESCRIPTION'
    self.lookahead = None
    self.ignore_trailer = False

  def Collect(self, strip_headings=False):
    """Returns the heading and content lines from text."""
    content = []
    if self.lookahead:
      if not strip_headings:
        content.append(self.lookahead)
      self.lookahead = None
    heading = self.heading
    self.heading = None
    while self.text:
      line = self.text.pop(0)
      usage = 'Usage:'
      if line.startswith(usage):
        line = line[len(usage):].strip()
        if line:
          self.lookahead = line
        self.heading = 'USAGE'
        break
      if line.endswith(':'):
        if 'Commands' in line:
          self.heading = 'COMMANDS'
          break
        if 'Examples' in line:
          self.heading = 'EXAMPLES'
          break
        if 'Options' in line:
          self.heading = 'FLAGS'
          break
      if line.startswith(' ') or not strip_headings and not self.ignore_trailer:
        content.append(line.rstrip())
    while content and not content[0]:
      content.pop(0)
    while content and not content[-1]:
      content.pop()
    self.ignore_trailer = True
    return heading, content


class KubectlCliTreeGenerator(CliTreeGenerator):
  """kubectl CLI tree generator."""

  def AddFlags(self, command, content, is_global=False):
    """Adds flags in content lines to command."""
    for line in content:
      flags, description = line.strip().split(':', 1)
      flag = flags.split(', ')[-1]
      name, value = flag.split('=')
      if value in ('true', 'false'):
        value = ''
        type_ = 'bool'
      else:
        value = 'VALUE'
        type_ = 'string'
      default = ''
      command[cli_tree.LOOKUP_FLAGS][name] = _Flag(
          name=name,
          description=description,
          type_=type_,
          value=value,
          default=default,
          is_required=False,
          is_global=is_global,
      )

  def SubTree(self, path):
    """Generates and returns the CLI subtree rooted at path."""
    command = _Command(path)
    text = self.Run(path + ['--help'])
    collector = _KubectlCollector(text)

    while True:
      heading, content = collector.Collect()
      if not heading:
        break
      elif heading == 'COMMANDS':
        for line in content:
          try:
            name = line.split()[0]
          except IndexError:
            continue
          command[cli_tree.LOOKUP_IS_GROUP] = True
          command[cli_tree.LOOKUP_COMMANDS][name] = self.SubTree(path + [name])
      elif heading in ('DESCRIPTION', 'EXAMPLES'):
        command[cli_tree.LOOKUP_SECTIONS][heading] = '\n'.join(content)
      elif heading == 'FLAGS':
        self.AddFlags(command, content)
    return command

  def GetVersion(self):
    """Returns the CLI_VERSION string."""
    if not self._cli_version:
      verbose_version = self.Run([self.cli_name, 'version', '--client'])
      match = re.search('GitVersion:"([^"]*)"', verbose_version)
      self._cli_version = match.group(1)
    return self._cli_version

  def GenerateTree(self):
    """Generates and returns the CLI tree rooted at path."""

    # Construct the tree minus the global flags.
    tree = self.SubTree([self.cli_name])

    # Add the global flags to the root.
    text = self.Run([self.cli_name, 'options'])
    collector = _KubectlCollector(text)
    _, content = collector.Collect(strip_headings=True)
    content.append('  --help=true: List detailed command help.')
    self.AddFlags(tree, content, is_global=True)

    # Finally add the version stamps.
    tree[cli_tree.LOOKUP_CLI_VERSION] = self.GetVersion()
    tree[cli_tree.LOOKUP_VERSION] = cli_tree.VERSION

    return tree


GENERATORS = {
    'bq': BqCliTreeGenerator,
    'gsutil': GsutilCliTreeGenerator,
    'kubectl': KubectlCliTreeGenerator,
}


def GetCliTreeGenerator(cli_name):
  """Returns the CLI tree generator for cli_name.

  Args:
    cli_name: The CLI root command name.

  Raises:
    NoCliTreeGeneratorForCommand: if cli_name does not have a CLI tree generator

  Returns:
    The CLI tree generator for cli_name.
  """
  try:
    return GENERATORS[cli_name](cli_name)
  except IndexError:
    raise NoCliTreeGeneratorForCommand(
        'No CLI tree generator for [{}].'.format(cli_name))


def GenerateCliTree(cli_name):
  """Generates and returns the CLI tree for cli_name.

  Args:
    cli_name: The CLI root command name. The 'help' subcommand, and subcommands
      with the '--help' flag, are run to discover the CLI tree structure and
      flags, positionals and help doc snippets.

  Raises:
    NoCliTreeGeneratorForCommand: if cli_name does not have a CLI tree generator

  Returns:
    The CLI tree for cli_name.
  """
  return GetCliTreeGenerator(cli_name).GenerateTree()


def IsCliTreeUpToDate(cli_name, tree):
  """Returns True if the CLI tree for cli_name is up to date.

  Args:
    cli_name: The CLI root command name.
    tree: The loaded CLI tree.

  Returns:
    True if the CLI tree for cli_name is up to date.
  """
  version = GetCliTreeGenerator(cli_name).GetGetVersion()
  return tree.get(cli_tree.LOOKUP_CLI_VERSION) == version


def UpdateCliTrees(cli=None, commands=None, directory=None,
                   verbose=False, warn_on_exceptions=False):
  """(re)generates the CLI trees in directory if non-existent or out ot date.

  This function uses the progress tracker because some of the updates can
  take ~minutes.

  Args:
    cli: The default CLI. If not None then the default CLI is also updated.
    commands: Update only the commands in this list.
    directory: The directory containing the CLI tree JSON files. If None
      then the default installation directory is used.
    verbose: Display a status line for up to date CLI trees if True.
    warn_on_exceptions: Emits warning messages in lieu of exceptions. Used
      during installation.

  Raises:
    NoCliTreeGeneratorForCommand: A command in commands is not supported
      (doesn't have a generator).
  """
  if not directory:
    try:
      directory = cli_tree.CliTreeDir()
    except cli_tree.SdkRootNotFoundError as e:
      if not warn_on_exceptions:
        raise
      log.warn(str(e))
  if commands:
    commands = set(commands)
  else:
    commands = None
  cli_name = cli_tree.DEFAULT_CLI_NAME
  if cli and (not commands or cli_name in commands):
    cli_tree.Load(cli=cli, verbose=verbose)
    if commands:
      commands.remove(cli_name)
  for cli_name, generator_class in sorted(GENERATORS.iteritems()):
    if commands is not None:
      if cli_name not in commands:
        continue
      commands.remove(cli_name)
    generator = generator_class(cli_name)
    if not generator.CliCommandExists():
      continue
    up_to_date = False
    path = os.path.join(directory, cli_name) + '.json'
    try:
      f = open(path, 'r')
    except IOError:
      pass  # We'll warn below.
    else:
      with f:
        try:
          tree = json.load(f)
        except ValueError:
          # Corrupt JSON -- could have been interrupted.
          tree = None
        if tree:
          version = generator.GetVersion()
          up_to_date = tree.get(cli_tree.LOOKUP_CLI_VERSION) == version
    if up_to_date:
      if verbose:
        log.status.Print('[{}] CLI tree version [{}] is up to date.'.format(
            cli_name, version))
      continue
    with progress_tracker.ProgressTracker(
        'Generating and updating the [{}] CLI tree'.format(cli_name)):
      tree = generator.GenerateTree()
      try:
        f = open(path, 'w')
      except IOError as e:
        if not warn_on_exceptions:
          raise
        log.warn(str(e))
      else:
        with f:
          resource_printer.Print(tree, print_format='json', out=f)
  if commands:
    raise NoCliTreeGeneratorForCommand('No CLI generators for [{}].'.format(
        ', '.join(sorted(commands))))
