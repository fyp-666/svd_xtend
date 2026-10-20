# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local compatibility fix for Diffusers 0.24 temporal feed-forward chunking."""

from typing import Optional

import torch
from diffusers.models.attention import (
    TemporalBasicTransformerBlock as DiffusersTemporalBasicTransformerBlock,
    _chunked_feed_forward,
)


class TemporalBasicTransformerBlock(DiffusersTemporalBasicTransformerBlock):
    """Preserve the checkpoint layout and use ff_in in both entry-FFN paths.

    Diffusers 0.24's chunked entry path calls ``ff`` instead of ``ff_in``.
    Enabling memory-saving chunking must not substitute a different learned
    layer. The constructor and all parameter names remain inherited unchanged.
    """

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        num_frames: int,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        batch_frames, seq_length, channels = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states = hidden_states.reshape(batch_size, num_frames, seq_length, channels)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size * seq_length, num_frames, channels)

        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)
        if self._chunk_size is not None:
            hidden_states = _chunked_feed_forward(self.ff_in, hidden_states, self._chunk_dim, self._chunk_size)
        else:
            hidden_states = self.ff_in(hidden_states)
        if self.is_res:
            hidden_states = hidden_states + residual

        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
        hidden_states = attn_output + hidden_states

        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm3(hidden_states)
        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states if self.is_res else ff_output

        hidden_states = hidden_states.reshape(batch_size, seq_length, num_frames, channels)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        return hidden_states.reshape(batch_size * num_frames, seq_length, channels)
