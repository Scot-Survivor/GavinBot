import re
import discord
import discord.utils
import asyncio
import datetime as dt
import concurrent.futures
from discord.ext import commands
from DatabaseTools import tool as database
from random import choice
from backend import load_model, predict


# 😂

class Gavin(commands.Cog):
    def __init__(self, bot: commands.Bot):
        database.connect()
        self.bot = bot
        self.archive_id = 785539080062631967
        self.loading = True
        self.START_TOKEN, self.END_TOKEN, self.tokenizer, self.MAX_LENGTH, self.model, self.ModelName, self.hparams = load_model(
            "../bunchOfLogs/" + input("Please enter model: "))
        self.swear_words = ['cock', 'tf', 'reggin', 'bellend', 'twat',
                            'bollocks', 'wtf', 'slag', 'fucker', 'rapist',
                            'shit', 'bitch', 'minger', 'nigger', 'fking',
                            'wanker', 'hentai', 'ffs', 'porn', 'tits',
                            'fucking', 'knob', 'minge', 'clunge', 'whore',
                            'bloodclat', 'fuck', 'cunt', 'crap', 'pissed',
                            'prick', 'nickger', 'cocks', 'pussy', "fucking",
                            "bullshit", "slut", "fuckin'", "slut"]

        self.loading = False
        self.phrases = ["My brain is in confinement until further notice",
                        "My brain is currently unavailable, please leave a message so I can ignore it, kthxbye"]

    @commands.Cog.listener()
    async def on_message(self, user_message: discord.Message):
        if not self.loading:
            await self.bot.change_presence(activity=discord.Game(name=f"Loaded Model {self.ModelName}"))
            if user_message.author != self.bot.user:
                pattern = re.compile(r"[^a-zA-Z?.!,'\"<>0-9 ]+")
                message = re.sub(pattern, "", user_message.content)
                words = message.split(' ')
                if words[0] == "<!753611486999478322>" or words[0] == "<753611486999478322>":
                    formatted_message = [words[1:]]
                    formatted_message = formatted_message[0]
                    message_for_bot = " ".join(formatted_message)
                    await self.chat(user_message, message_for_bot)
        else:
            if user_message.author != self.bot.user:
                channel_id = int(user_message.channel.id)
                channel = discord.utils.get(user_message.guild.text_channels, id=channel_id)
                channel.send(choice(self.phrases))

    async def chat(self, message: discord.Message, content: str):
        channel_id = int(message.channel.id)
        channel = discord.utils.get(message.guild.text_channels, id=channel_id)
        with channel.typing():
            response = predict(content, self.tokenizer, self.swear_words, self.START_TOKEN, self.END_TOKEN,
                               self.MAX_LENGTH, self.model)
            await database.sql_insert_into(str(message.guild.id), str(message.channel.id), self.ModelName,
                                           message.author,
                                           message.content, response,
                                           date=dt.datetime.now().strftime('%d/%m/%Y %H-%M-%S.%f')[:-2])

            msg = f"> {content}\n {message.author.mention} {response}"
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
                # await sent.add_reaction(emoji="😂")
        '''
        while user != self.bot.user:
            def check(r, u):
                return r, u

            try:  # TODO Fix this logic
                u_reaction, user = await self.bot.wait_for('reaction_add', timeout=60.0, check=check)
            except asyncio.TimeoutError:
                return
            else:
                if u_reaction.emoji == "😂" and user != sent.author:
                    archive_channel = discord.utils.get(message.guild.text_channels, id=self.archive_id)
                    await archive_channel.send(msg)
                    await channel.send("Added to archives!")
                    return
        '''
        return

    @commands.command(name="hparams", aliases=['params', 'hp'])
    async def hparams(self, ctx: commands.Context):
        """Display the hparams of the mode. Aliases: params and hp"""
        fields = ["Samples", "Max Length Of Words", "Batch Size", "Buffer Size", "Layers",
                  "d_model", "Heads", "Units", "Dropout", "Vocab Size", "Target Vocab Size"]
        embed = discord.Embed(title=f"Hparams for {self.ModelName}", type="rich",
                              description="Displays the Hyper Parameters used for the bot")
        for i, value in enumerate(self.hparams):
            embed.add_field(name=f"{fields[i]}", value=f"{value}", inline=False)
        await ctx.send(embed=embed)

    @commands.command(name="reload", aliases=['r'])
    async def reload_model(self, ctx: commands.Context, model_name):
        """Reloads the model. !reload <model name>. Alias: r"""
        if ctx.message.author.id == 348519271460110338:
            self.loading = True
            await self.bot.change_presence(activity=discord.Game(name=f"Loading new model {model_name}"))
            with concurrent.futures.ThreadPoolExecutor(1) as executor:
                future = executor.submit(load_model, "../bunchOfLogs/" + model_name)
                self.START_TOKEN, self.END_TOKEN, self.tokenizer, self.MAX_LENGTH, self.model, self.ModelName, self.hparams = future.result()
            self.loading = False
            await self.bot.change_presence(activity=discord.Game(name=f"Loaded into {model_name}"))

    @commands.command(name="image", aliases=['img', 'im'])
    async def send_image(self, ctx: commands.Context):
        """Send the image of what the models Layers look like. Alias: img and im"""
        try:
            with open(f"../bunchOfLogs/{self.ModelName}/images/{self.ModelName}_Image.png", "rb") as f:
                picture = discord.File(f)
        except Exception as e:
            await ctx.send(f"Error on image send: {e}")
        else:
            await ctx.send(file=picture)

    @commands.command(name="invite", aliases=['inv'])
    async def send_invite(self, ctx: commands.Context):
        """Send the bots invite link"""
        await ctx.send(f"You can add me here!\nhttps://discord.com/api/oauth2/authorize?client_id=753611486999478322&permissions=378944&scope=bot")


def setup(bot):
    bot.add_cog(Gavin(bot))
