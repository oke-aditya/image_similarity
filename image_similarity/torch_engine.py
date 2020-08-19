"""
I can write this if we need custom training loop etc.
I usually use this in PyTorch.
"""

__all__ = ["train_step", "val_step"]

import torch
import torch.nn as nn


def train_step(encoder, decoder, train_loader, loss_fn, optimizer, device):

    encoder.train()
    decoder.train()

    for batch_idx, (train_img, target_img) in enumerate(train_loader):
        train_img.to(device)
        target_img.to(device)

        optimizer.zero_grad()

        enc_output = encoder(train_img)
        dec_output = decoder(enc_output)

        loss = loss_fn(dec_output, target_img)
        loss.backward()

        optimizer.step()

    return loss.item()


def val_step(encoder, decoder, val_loader, loss_fn, device):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for batch_idx, (train_img, target_img) in enumerate(train_loader):
            train_img.to(device)
            target_img.to(device)

            enc_output = encoder(train_img)
            dec_output = decoder(enc_output)

            loss = loss_fn(enc_output, dec_output)

    return loss.item()
