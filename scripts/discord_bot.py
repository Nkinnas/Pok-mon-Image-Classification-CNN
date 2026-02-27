
import io
import json
import os
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands
import torch
from dotenv import load_dotenv
from PIL import Image
from torchvision import models, transforms

load_dotenv()

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("DISCORD_BOT_TOKEN not set. Create a .env file with your bot token.")

# MODEL

ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = ROOT / "checkpoints" / "best.pt"
IDX_PATH = ROOT / "checkpoints" / "idx_to_class.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(IDX_PATH, "r") as f:
    idx_to_class = json.load(f)
num_classes = len(idx_to_class)

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
model.load_state_dict(ckpt["model_state"])
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def get_class_name(idx):
    return idx_to_class.get(str(idx)) or idx_to_class.get(idx)


def predict(img: Image.Image):
    """Return list of (class_name, confidence%) sorted by confidence."""
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0)
    top3 = torch.topk(probs, k=3)
    return [
        (get_class_name(idx.item()), prob.item() * 100)
        for prob, idx in zip(top3.values, top3.indices)
    ]


def build_embed(results, image_url=None):
    """Build a Discord embed from prediction results."""
    name, conf = results[0]
    color = discord.Color.green() if conf >= 70 else (
        discord.Color.gold() if conf >= 40 else discord.Color.red()
    )

    embed = discord.Embed(title=f"It's {name}!", color=color)
    if image_url:
        embed.set_thumbnail(url=image_url)

    ranking = "\n".join(
        f"**{i}. {n}** — {c:.1f}%" for i, (n, c) in enumerate(results, 1)
    )
    embed.add_field(name="Top Predictions", value=ranking, inline=False)
    embed.set_footer(text="Pokemon CNN  |  ResNet-18  |  Gen 1")
    return embed


#DISCORD BOT

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def image_attachment(message: discord.Message):
    """Return the first image attachment on a message, or None."""
    for att in message.attachments:
        if Path(att.filename).suffix.lower() in IMG_EXTS:
            return att
    return None


async def run_predict_attachment(attachment: discord.Attachment):
    """Download an attachment, run inference, return (results, embed)."""
    img_bytes = await attachment.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    results = predict(img)
    embed = build_embed(results, image_url=attachment.url)
    return embed


#Slash command: /predict with image attachment

@bot.tree.command(name="predict", description="Upload a Pokemon image and get a prediction")
@app_commands.describe(image="A Pokemon image to identify")
async def slash_predict(interaction: discord.Interaction, image: discord.Attachment):
    if Path(image.filename).suffix.lower() not in IMG_EXTS:
        await interaction.response.send_message("That doesn't look like an image file.", ephemeral=True)
        return

    await interaction.response.defer()
    embed = await run_predict_attachment(image)
    await interaction.followup.send(embed=embed)


#! command: !predict with image attachment

@bot.command(name="predict")
async def prefix_predict(ctx: commands.Context):
    attachment = image_attachment(ctx.message)
    if attachment is None:
        await ctx.reply("Attach an image to your message and try again!")
        return
    async with ctx.typing():
        embed = await run_predict_attachment(attachment)
    await ctx.reply(embed=embed)


#Bot mention 

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    #let commands run first
    await bot.process_commands(message)

    #Bot respond with image and
    if bot.user in message.mentions:
        attachment = image_attachment(message)
        if attachment:
            async with message.channel.typing():
                embed = await run_predict_attachment(attachment)
            await message.reply(embed=embed)


#Bot startup

@bot.event
async def on_ready():
    synced = await bot.tree.sync()
    print(f"Logged in as {bot.user} (id: {bot.user.id})")
    print(f"Synced {len(synced)} slash command(s)")
    print(f"Model loaded on {device} — {num_classes} classes")


bot.run(TOKEN)
