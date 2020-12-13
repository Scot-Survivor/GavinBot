from discord.ext import commands
from DatabaseTools import tool
from backend import predict, ModelName, hparams
import re
import discord
import discord.utils
import asyncio
import datetime as dt


# 😂

class Gavin(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.connection, self.c = tool.connect()
        self.bot = bot
        self.archive_id = 784517999255486515

    @commands.Cog.listener()
    async def on_message(self, user_message: discord.Message):
        if user_message.author != self.bot.user:
            pattern = re.compile(r"[^a-zA-Z?.!,'<>0-9 ]+")
            message = re.sub(pattern, "", user_message.content)
            words = message.split(' ')
            if words[0] == "<!753611486999478322>" or words[0] == "<753611486999478322>":
                formatted_message = [words[1:]]
                formatted_message = formatted_message[0]
                message_for_bot = " ".join(formatted_message)
                await self.chat(user_message, message_for_bot)

    async def chat(self, message: discord.Message, content: str):
        channel_id = int(message.channel.id)
        channel = discord.utils.get(message.guild.text_channels, id=channel_id)
        with channel.typing():
            response = predict(content)
            await tool.sql_insert_into(str(message.guild.id), str(message.channel.id), ModelName, message.author,
                                       message.content, response,
                                       date=dt.datetime.now().strftime('%d/%m/%Y %H-%M-%S.%f')[:-2],
                                       u_connection=self.connection, cursor=self.c)
            user = "<@!"
            user += str(message.author.id) + ">"
            msg = f"> {content}\n {user} {response}"
            print(f"""Date: {dt.datetime.now().strftime('%d/%m/%Y %H-%M-%S.%f')[:-2]}
Author: {message.author}
Where: {message.guild}:{message.channel}
Input: {content}
Output: {response}""")
            if response is None or response == "":
                await channel.send("👎")
                return
            else:
                sent = await channel.send(msg)
                user = None
                u_reaction = None
                await sent.add_reaction(emoji="😂")

        while user != self.bot.user:
            def check(r, u):
                return r, u

            try:
                u_reaction, user = await self.bot.wait_for('reaction_add', timeout=60.0, check=check)
            except asyncio.TimeoutError:
                return
            else:
                if u_reaction.emoji == "😂" and user != self.bot.user and user != sent.author:
                    archive_channel = discord.utils.get(message.guild.text_channels, id=self.archive_id)
                    await archive_channel.send(msg)
                    await channel.send("Added to archives!")
                else:
                    return
        return

    @commands.command(name="hparams", aliases=['params', 'hp'])
    async def hparams(self, ctx: commands.Context):
        fields = ["Samples", "Max Length Of Words", "Batch Size", "Buffer Size", "Layers",
                  "d_model", "Heads", "Units", "Dropout", "Vocab Size", "Target Vocab Size"]
        embed = discord.Embed(title=f"Hparams for {ModelName}", type="rich",
                              description="Displays the Hyper Parameters used for the bot")
        for i, value in enumerate(hparams):
            embed.add_field(name=f"{fields[i]}", value=f"{value}", inline=False)
        await ctx.send(embed=embed)


def setup(bot):
    bot.add_cog(Gavin(bot))
