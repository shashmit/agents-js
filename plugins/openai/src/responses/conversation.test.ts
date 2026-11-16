// SPDX-FileCopyrightText: 2026 LiveKit, Inc.
//
// SPDX-License-Identifier: Apache-2.0
import { ChatContext, initializeLogger, llm } from '@livekit/agents';
import { describe, expect, it } from 'vitest';
import {
  ConversationState,
  INTERRUPTED_TOOL_OUTPUT,
  MAX_TOOL_OUTPUT_RECOVERIES,
  parseMissingToolOutputCallId,
} from './conversation.js';

initializeLogger({ level: 'silent', pretty: false });

function baseCtx(): ChatContext {
  const ctx = ChatContext.empty();
  ctx.addMessage({ role: 'system', content: 'You are a helpful assistant.' });
  ctx.addMessage({ role: 'user', content: 'hi' });
  return ctx;
}

/** Runs one turn to completion, as if the model streamed `text` and called `callIds`. */
async function completeTurn(
  state: ConversationState,
  ctx: ChatContext,
  { text = '', callIds = [] as string[] } = {},
) {
  const turn = state.beginTurn(ctx);
  const input = await turn.toInput();
  if (text) turn.recordText(text);
  for (const callId of callIds) turn.recordFunctionCall(callId);
  turn.commit();
  return { turn, input };
}

describe('parseMissingToolOutputCallId', () => {
  it('extracts the call id from the OpenAI error message', () => {
    expect(
      parseMissingToolOutputCallId('No tool output found for function call call_AbC-12_x.'),
    ).toBe('call_AbC-12_x');
  });

  it('ignores unrelated errors', () => {
    expect(parseMissingToolOutputCallId('Rate limit reached')).toBeUndefined();
    expect(parseMissingToolOutputCallId(undefined)).toBeUndefined();
  });
});

describe('ConversationState', () => {
  it('sends system messages as instructions instead of conversation items', async () => {
    const state = new ConversationState('conv_1');
    const ctx = baseCtx();
    ctx.addMessage({ role: 'developer', content: 'Greet the user.' });

    const turn = state.beginTurn(ctx);

    expect(turn.conversationId).toBe('conv_1');
    expect(turn.instructions).toBe('You are a helpful assistant.\n\nGreet the user.');
    expect(await turn.toInput()).toEqual([{ role: 'user', content: 'hi' }]);
  });

  it('only sends items the conversation does not hold yet', async () => {
    const state = new ConversationState('conv_1');
    const ctx = baseCtx();
    await completeTurn(state, ctx, { text: 'Hello! How can I help?' });

    ctx.addMessage({ role: 'assistant', content: 'Hello! How can I help?' });
    ctx.addMessage({ role: 'user', content: 'what is my history?' });

    expect(await state.beginTurn(ctx).toInput()).toEqual([
      { role: 'user', content: 'what is my history?' },
    ]);
  });

  it('skips an assistant message truncated by an interruption', async () => {
    const state = new ConversationState('conv_1');
    const ctx = baseCtx();
    await completeTurn(state, ctx, { text: 'Hello! How can I help you today?' });

    ctx.addMessage({ role: 'assistant', content: 'Hello! How can', interrupted: true });
    ctx.addMessage({ role: 'user', content: 'stop' });

    expect(await state.beginTurn(ctx).toInput()).toEqual([{ role: 'user', content: 'stop' }]);
  });

  it('sends assistant messages the model did not produce (e.g. session.say)', async () => {
    const state = new ConversationState('conv_1');
    const ctx = baseCtx();
    await completeTurn(state, ctx, { text: 'Hello!' });

    ctx.addMessage({ role: 'assistant', content: 'Hello!' });
    ctx.addMessage({ role: 'assistant', content: 'Please hold on.' });
    ctx.addMessage({ role: 'user', content: 'ok' });

    expect(await state.beginTurn(ctx).toInput()).toEqual([
      { role: 'assistant', content: 'Please hold on.' },
      { role: 'user', content: 'ok' },
    ]);
  });

  it('sends a tool output whose call is stored server-side', async () => {
    const state = new ConversationState('conv_1');
    const ctx = baseCtx();
    await completeTurn(state, ctx, { callIds: ['call_1'] });

    ctx.insert([
      llm.FunctionCall.create({ callId: 'call_1', name: 'get_history', args: '{}' }),
      llm.FunctionCallOutput.create({ callId: 'call_1', output: 'none', isError: false }),
    ]);

    // ChatContext.toProviderFormat would drop this output because its call is not in the input.
    expect(await state.beginTurn(ctx).toInput()).toEqual([
      { type: 'function_call_output', call_id: 'call_1', output: 'none' },
    ]);
  });

  it('answers a stored call that never got an output before anything else', async () => {
    const state = new ConversationState('conv_1');
    const ctx = baseCtx();
    await completeTurn(state, ctx, { callIds: ['call_1'] });

    ctx.addMessage({ role: 'user', content: 'never mind' });
    const { input } = await completeTurn(state, ctx);

    expect(input).toEqual([
      { type: 'function_call_output', call_id: 'call_1', output: INTERRUPTED_TOOL_OUTPUT },
      { role: 'user', content: 'never mind' },
    ]);

    // Once answered, the placeholder is not sent again.
    ctx.addMessage({ role: 'user', content: 'next' });
    expect(await state.beginTurn(ctx).toInput()).toEqual([{ role: 'user', content: 'next' }]);
  });

  it('places a placeholder right after a local call that has no output', async () => {
    const state = new ConversationState('conv_1');
    const ctx = baseCtx();
    ctx.insert(llm.FunctionCall.create({ callId: 'call_2', name: 'lookup', args: '{}' }));

    expect(await state.beginTurn(ctx).toInput()).toEqual([
      { role: 'user', content: 'hi' },
      { type: 'function_call', call_id: 'call_2', name: 'lookup', arguments: '{}' },
      { type: 'function_call_output', call_id: 'call_2', output: INTERRUPTED_TOOL_OUTPUT },
    ]);
  });

  it('resends the input of a turn that never completed', async () => {
    const state = new ConversationState('conv_1');
    const ctx = baseCtx();
    const abandoned = state.beginTurn(ctx);
    await abandoned.toInput();

    ctx.addMessage({ role: 'user', content: 'are you there?' });

    expect(await state.beginTurn(ctx).toInput()).toEqual([
      { role: 'user', content: 'hi' },
      { role: 'user', content: 'are you there?' },
    ]);
  });

  it('patches calls reported missing by the server within a bounded budget', async () => {
    const state = new ConversationState('conv_1');
    const turn = state.beginTurn(baseCtx());

    expect(turn.recoverMissingToolOutput('call_old')).toBe(true);
    expect(turn.recoverMissingToolOutput('call_old')).toBe(false);
    expect((await turn.toInput())[0]).toEqual({
      type: 'function_call_output',
      call_id: 'call_old',
      output: INTERRUPTED_TOOL_OUTPUT,
    });

    for (let i = 1; i < MAX_TOOL_OUTPUT_RECOVERIES; i++) {
      expect(turn.recoverMissingToolOutput(`call_${i}`)).toBe(true);
    }
    expect(turn.recoverMissingToolOutput('call_over_budget')).toBe(false);
  });
});
