// SPDX-FileCopyrightText: 2025 LiveKit, Inc.
//
// SPDX-License-Identifier: Apache-2.0
import {
  APIStatusError,
  ChatContext,
  ConnectionPool,
  initializeLogger,
  stream,
} from '@livekit/agents';
import { llm, llmStrict } from '@livekit/agents-plugins-test';
import { EventEmitter } from 'node:events';
import { describe, expect, it, vi } from 'vitest';
import { LLM } from '../responses/llm.js';
import { ResponsesWebSocket, WSLLM, WSLLMStream, buildResponsesWsUrl } from './llm.js';
import type { WsResponseCreateEvent, WsServerEvent } from './types.js';
import { wsServerEventSchema } from './types.js';

initializeLogger({ level: 'silent', pretty: false });

const hasOpenAIApiKey = Boolean(process.env.OPENAI_API_KEY);

describe('buildResponsesWsUrl', () => {
  it('points at the OpenAI Responses WS endpoint without model when no baseURL is set', () => {
    const url = new URL(buildResponsesWsUrl(undefined, 'gpt-4.1'));

    expect(url.protocol).toBe('wss:');
    expect(url.host).toBe('api.openai.com');
    expect(url.pathname).toBe('/v1/responses');
    expect(url.searchParams.get('model')).toBe(null);
  });

  it('rewrites https baseURL to wss and appends /responses with the model', () => {
    const url = new URL(buildResponsesWsUrl('https://gateway.example.com/v1', 'gpt-4o'));

    expect(url.protocol).toBe('wss:');
    expect(url.host).toBe('gateway.example.com');
    expect(url.pathname).toBe('/v1/responses');
    expect(url.searchParams.get('model')).toBe('gpt-4o');
  });

  it('rewrites full https Responses endpoint to wss without duplicating /responses', () => {
    const url = new URL(buildResponsesWsUrl('https://gateway.example.com/v1/responses', 'gpt-4o'));

    expect(url.protocol).toBe('wss:');
    expect(url.host).toBe('gateway.example.com');
    expect(url.pathname).toBe('/v1/responses');
    expect(url.searchParams.get('model')).toBe('gpt-4o');
  });

  it('uses full wss Responses endpoint with the model', () => {
    const url = new URL(buildResponsesWsUrl('wss://gateway.example.com/v1/responses', 'gpt-4o'));

    expect(url.protocol).toBe('wss:');
    expect(url.host).toBe('gateway.example.com');
    expect(url.pathname).toBe('/v1/responses');
    expect(url.searchParams.get('model')).toBe('gpt-4o');
  });

  it('omits the model when an explicit baseURL still points at api.openai.com', () => {
    const url = new URL(buildResponsesWsUrl('https://api.openai.com/v1', 'gpt-4.1'));

    expect(url.host).toBe('api.openai.com');
    expect(url.pathname).toBe('/v1/responses');
    expect(url.searchParams.get('model')).toBe(null);
  });

  it('strips a trailing slash on baseURL before appending /responses', () => {
    const url = new URL(buildResponsesWsUrl('https://gateway.example.com/v1/', 'gpt-4o-mini'));

    expect(url.pathname).toBe('/v1/responses');
    expect(url.searchParams.get('model')).toBe('gpt-4o-mini');
  });

  it('rewrites http baseURL to ws (not wss)', () => {
    const url = new URL(buildResponsesWsUrl('http://gateway.example.com/v1', 'gpt-4o-mini'));

    expect(url.protocol).toBe('ws:');
    expect(url.host).toBe('gateway.example.com');
    expect(url.pathname).toBe('/v1/responses');
    expect(url.searchParams.get('model')).toBe('gpt-4o-mini');
  });

  it('strips a trailing slash on an http baseURL before appending /responses', () => {
    const url = new URL(buildResponsesWsUrl('http://gateway.example.com/v1/', 'gpt-4o-mini'));

    expect(url.protocol).toBe('ws:');
    expect(url.pathname).toBe('/v1/responses');
    expect(url.searchParams.get('model')).toBe('gpt-4o-mini');
  });
});

// opens every stream, on every provider
const RESPONSE_CREATED: WsServerEvent = {
  type: 'response.created',
  response: { id: 'resp_1' },
};

// a reasoning phase marker: a message item with no text of its own
const PHASE_METADATA: WsServerEvent = {
  type: 'response.output_item.done',
  item: { type: 'message', phase: 'thinking' },
};

const TEXT_DELTA: WsServerEvent = {
  type: 'response.output_text.delta',
  delta: 'hello',
};

function responseIncomplete(reason?: string): WsServerEvent {
  return {
    type: 'response.incomplete',
    response: {
      id: 'resp_1',
      incomplete_details: reason === undefined ? {} : { reason },
    },
  };
}

describe('OpenAI Responses WS incomplete events', () => {
  it.each(['max_output_tokens', 'content_filter'])('parses the provider reason %s', (reason) => {
    const parsed = wsServerEventSchema.parse(responseIncomplete(reason));

    expect(parsed.type).toBe('response.incomplete');
    if (parsed.type !== 'response.incomplete') throw new Error('expected incomplete event');
    expect(parsed.response.incomplete_details?.reason).toBe(reason);
  });

  it('accepts incomplete details without a reason', () => {
    const parsed = wsServerEventSchema.parse({
      type: 'response.incomplete',
      response: { id: 'resp_1', incomplete_details: {} },
    });

    expect(parsed.type).toBe('response.incomplete');
    if (parsed.type !== 'response.incomplete') throw new Error('expected incomplete event');
    expect(parsed.response.incomplete_details?.reason).toBeUndefined();
  });

  it('closes the request channel after the terminal frame', async () => {
    class RecordingSocket extends EventEmitter {
      readyState = 1;
      sent: string[] = [];

      send(data: string): void {
        this.sent.push(data);
      }

      close(): void {}
    }

    const socket = new RecordingSocket();
    const transport = new ResponsesWebSocket(socket as never);
    const channel = transport.sendRequest({
      type: 'response.create',
      model: 'gpt-4.1',
      input: [],
    });

    socket.emit('message', Buffer.from(JSON.stringify(responseIncomplete())));
    const events = [];
    for await (const event of channel.stream()) events.push(event);

    expect(events).toEqual([responseIncomplete()]);
    expect(socket.sent).toHaveLength(1);
  });
});

/** Replays the given frames, then dies the way a socket going quiet does. */
class StallingConnection {
  attempts = 0;

  constructor(private readonly frames: WsServerEvent[]) {}

  sendRequest(): stream.StreamChannel<WsServerEvent> {
    this.attempts++;
    const channel = stream.createStreamChannel<WsServerEvent>();

    void (async () => {
      for (const frame of this.frames) {
        await channel.write(frame);
      }
      await channel.abort(new Error('stalled mid-stream'));
    })();

    return channel;
  }

  close(): void {}
}

/** Returns how many times the response was requested before the stream gave up. */
async function attemptsUntilStall(frames: WsServerEvent[], maxRetry: number): Promise<number> {
  const wsLLM = new WSLLM({ model: 'gpt-4.1', apiKey: 'test-key' });
  // the stream emits 'error' on every failed attempt; without a listener the
  // EventEmitter turns it into an unhandled error and fails the run
  wsLLM.on('error', () => {});

  const conn = new StallingConnection(frames);
  const pool = new ConnectionPool<ResponsesWebSocket>({
    connectCb: async () => conn as unknown as ResponsesWebSocket,
  });

  const chatCtx = ChatContext.empty();
  chatCtx.addMessage({ role: 'user', content: 'hi' });

  const llmStream = new WSLLMStream(wsLLM, {
    pool,
    model: 'gpt-4.1',
    chatCtx,
    fullChatCtx: chatCtx,
    connOptions: { maxRetry, retryIntervalMs: 0, timeoutMs: 5000 },
    modelOptions: {},
    strictToolSchema: true,
  });

  try {
    for await (const _chunk of llmStream) {
      void _chunk;
    }
  } catch {
    // the stall surfaces through the 'error' event; the iterator just ends
  }

  return conn.attempts;
}

describe('OpenAI Responses WS retry eligibility', () => {
  it('retries a stall that follows the stream-opening frame alone', async () => {
    expect(await attemptsUntilStall([RESPONSE_CREATED], 2)).toBe(3);
  });

  it('retries a stall that follows a phase marker', async () => {
    expect(await attemptsUntilStall([RESPONSE_CREATED, PHASE_METADATA], 2)).toBe(3);
  });

  it('does not retry once generated text has reached the caller', async () => {
    expect(await attemptsUntilStall([RESPONSE_CREATED, TEXT_DELTA], 2)).toBe(1);
  });
});

describe('OpenAI Responses WS incomplete handling', () => {
  it.each([
    ['max_output_tokens', 'max_output_tokens'],
    [undefined, 'reason unavailable'],
  ] as const)(
    'surfaces %s as non-retryable without updating context',
    async (reason, expectedMessage) => {
      class ReplayConnection {
        attempts = 0;
        payloads: WsResponseCreateEvent[] = [];

        sendRequest(payload: WsResponseCreateEvent): stream.StreamChannel<WsServerEvent> {
          this.attempts++;
          this.payloads.push(payload);
          const channel = stream.createStreamChannel<WsServerEvent>();
          void (async () => {
            await channel.write(RESPONSE_CREATED);
            await channel.write(responseIncomplete(reason));
            await channel.close();
          })();
          return channel;
        }

        close(): void {}
      }

      const model = new WSLLM({ model: 'gpt-4.1', apiKey: 'test-key' });
      const onCompleted = vi.spyOn(model, '_onResponseCompleted');
      const errorEvent = new Promise<Error>((resolve) => {
        model.once('error', (event) => resolve(event.error));
      });
      const conn = new ReplayConnection();
      let connectCalls = 0;
      const pool = new ConnectionPool<ResponsesWebSocket>({
        connectCb: async () => {
          connectCalls++;
          return conn as unknown as ResponsesWebSocket;
        },
      });
      const chatCtx = ChatContext.empty();
      chatCtx.addMessage({ role: 'user', content: 'hi' });
      const llmStream = new WSLLMStream(model, {
        pool,
        model: 'gpt-4.1',
        chatCtx,
        fullChatCtx: chatCtx,
        connOptions: { maxRetry: 2, retryIntervalMs: 0, timeoutMs: 5000 },
        modelOptions: {},
        strictToolSchema: true,
      });

      for await (const _chunk of llmStream) void _chunk;
      const error = await errorEvent;

      expect(error).toBeInstanceOf(APIStatusError);
      expect(error.message).toContain(expectedMessage);
      expect((error as APIStatusError).statusCode).toBe(-1);
      expect((error as APIStatusError).retryable).toBe(false);
      expect(conn.attempts).toBe(1);
      expect(onCompleted).not.toHaveBeenCalled();

      await pool.withConnection(async (pooled) => expect(pooled).toBe(conn));
      expect(connectCalls).toBe(1);
      expect(conn.payloads[0]).not.toHaveProperty('previous_response_id');
      await pool.close();
      await model.aclose();
    },
  );
});

if (hasOpenAIApiKey) {
  describe('OpenAI Responses WS wrapper', async () => {
    await llm(
      new LLM({
        temperature: 0,
        strictToolSchema: false,
        useWebSocket: true,
      }),
      true,
    );
  });

  describe('OpenAI Responses WS wrapper strict tool schema', async () => {
    await llmStrict(
      new LLM({
        temperature: 0,
        strictToolSchema: true,
        useWebSocket: true,
      }),
    );
  });
} else {
  describe('OpenAI Responses WS wrapper', () => {
    it.skip('requires OPENAI_API_KEY', () => {});
  });
}

describe('OpenAI Responses WS conversation mode', () => {
  /** Answers each response.create with the next scripted frame sequence. */
  class ScriptedConnection {
    payloads: WsResponseCreateEvent[] = [];

    constructor(private readonly scripts: WsServerEvent[][]) {}

    sendRequest(payload: WsResponseCreateEvent): stream.StreamChannel<WsServerEvent> {
      const frames = this.scripts[this.payloads.length] ?? [];
      this.payloads.push(payload);
      const channel = stream.createStreamChannel<WsServerEvent>();
      void (async () => {
        for (const frame of frames) await channel.write(frame);
        await channel.close();
      })();
      return channel;
    }

    close(): void {}
  }

  const completed = (text: string): WsServerEvent[] => [
    RESPONSE_CREATED,
    { type: 'response.output_text.delta', delta: text },
    { type: 'response.completed', response: { id: 'resp_1' } },
  ];

  async function runTurn(model: WSLLM, conn: ScriptedConnection, chatCtx: ChatContext) {
    // route the stream through the scripted connection instead of a real socket
    const pool = new ConnectionPool<ResponsesWebSocket>({
      connectCb: async () => conn as unknown as ResponsesWebSocket,
    });
    vi.spyOn(ConnectionPool.prototype, 'withConnection').mockImplementationOnce((fn) =>
      pool.withConnection(fn as never),
    );
    let text = '';
    for await (const chunk of model.chat({ chatCtx })) text += chunk.delta?.content ?? '';
    await pool.close();
    return text;
  }

  it('sends only new items with conversation and instructions, never previous_response_id', async () => {
    const model = new WSLLM({ model: 'gpt-4.1', apiKey: 'test-key', conversation: 'conv_1' });
    const conn = new ScriptedConnection([completed('Hello!'), completed('Sure.')]);
    const chatCtx = ChatContext.empty();
    chatCtx.addMessage({ role: 'system', content: 'Be brief.' });
    chatCtx.addMessage({ role: 'user', content: 'hi' });

    expect(await runTurn(model, conn, chatCtx)).toBe('Hello!');
    chatCtx.addMessage({ role: 'assistant', content: 'Hello!' });
    chatCtx.addMessage({ role: 'user', content: 'help me' });
    await runTurn(model, conn, chatCtx);

    for (const payload of conn.payloads) {
      expect(payload.conversation).toBe('conv_1');
      expect(payload.instructions).toBe('Be brief.');
      expect(payload).not.toHaveProperty('previous_response_id');
    }
    expect(conn.payloads[0]!.input).toEqual([{ role: 'user', content: 'hi' }]);
    expect(conn.payloads[1]!.input).toEqual([{ role: 'user', content: 'help me' }]);
    await model.aclose();
  });

  it('answers a call the server reports as missing an output and resends', async () => {
    const model = new WSLLM({ model: 'gpt-4.1', apiKey: 'test-key', conversation: 'conv_1' });
    const missingOutput: WsServerEvent = {
      type: 'error',
      status: 400,
      error: { message: 'No tool output found for function call call_stale.' },
    };
    const conn = new ScriptedConnection([[missingOutput], completed('Back again.')]);
    const chatCtx = ChatContext.empty();
    chatCtx.addMessage({ role: 'user', content: 'hi' });

    expect(await runTurn(model, conn, chatCtx)).toBe('Back again.');
    expect(conn.payloads).toHaveLength(2);
    expect(conn.payloads[1]!.input).toEqual([
      {
        type: 'function_call_output',
        call_id: 'call_stale',
        output: 'The tool call was interrupted before a result was returned.',
      },
      { role: 'user', content: 'hi' },
    ]);
    await model.aclose();
  });
});
