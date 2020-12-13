import os
from discord.ext import tasks
from discord.ext.commands.errors import ExtensionFailed


class Extension:
    def __init__(self, name):
        self.name = name
        self.file = f'{name}.py'
        self.mtime = os.stat(self.file).st_mtime
        self.bot = None

    def has_changed(self):
        mtime = os.stat(self.file).st_mtime
        changed = mtime != self.mtime
        self.mtime = mtime
        return changed

    @tasks.loop(seconds=1.0)
    async def _check_change_task(self):
        assert self.bot is not None
        if self.has_changed():
            try:
                self.bot.reload_extension(self.name)
            except ExtensionFailed as e:
                print(f'unable to reload:\n{e}')
            else:
                print(f"Reloaded: {self.name} ")

    def bind_to(self, bot):
        self.bot = bot
        self.bot.load_extension(self.name)
        self._check_change_task.start()  # pylint: disable=no-member
