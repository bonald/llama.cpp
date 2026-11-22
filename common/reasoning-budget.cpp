#include "reasoning-budget.h"
#include "common.h"
#include "trie.h"
#include "unicode.h"

#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

static bool rb_loop_guard_enabled() {
    const char * env = std::getenv("LLAMA_REASONING_LOOP");
    if (!env || !*env) {
        return true;
    }
    return std::strcmp(env, "0") != 0 && std::strcmp(env, "off") != 0 && std::strcmp(env, "false") != 0;
}

static uint64_t rb_fnv1a(const std::string & s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 0x100000001b3ULL;
    }
    return h;
}

static bool rb_is_word_byte(unsigned char c) {
    return c >= 0x80 || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z') || c == '_';
}

struct reasoning_loop_tracker {
    static constexpr int32_t  kCheckTokens      = 256;
    static constexpr size_t   kShortWindowWords = 512;
    static constexpr size_t   kLongWindowWords  = 2000;
    static constexpr size_t   kPassageWords     = 12;
    static constexpr uint32_t kRepeats          = 3;
    static constexpr double   kShortCoverage    = 0.35;
    static constexpr double   kLongCoverage     = 0.25;
    static constexpr int32_t  kShortMaxTokens   = 2048;
    static constexpr uint32_t kShortConfirm     = 2;
    static constexpr size_t   kTailChars        = 1024;
    static constexpr size_t   kTailPeriod       = 64;
    static constexpr size_t   kMaxWords         = 30000;

    std::vector<uint64_t> words;
    std::string partial;
    std::string tail;
    int32_t  tokens_seen = 0;
    int32_t  next_check  = kCheckTokens;
    uint32_t short_hits  = 0;

    void reset() {
        words.clear();
        partial.clear();
        tail.clear();
        tokens_seen = 0;
        next_check  = kCheckTokens;
        short_hits  = 0;
    }

    void feed(const std::string & piece) {
        tokens_seen++;
        tail.append(piece);
        if (tail.size() > 4 * kTailChars) {
            tail.erase(0, tail.size() - 2 * kTailChars);
        }
        for (unsigned char c : piece) {
            if (rb_is_word_byte(c)) {
                partial.push_back((c >= 'A' && c <= 'Z') ? char(c - 'A' + 'a') : char(c));
            } else {
                if (!partial.empty()) {
                    words.push_back(rb_fnv1a(partial));
                    partial.clear();
                }
                if (c > ' ') {
                    std::string punct(1, char(c));
                    words.push_back(rb_fnv1a(punct));
                }
            }
        }
        if (words.size() > 2 * kMaxWords) {
            words.erase(words.begin(), words.end() - kMaxWords);
        }
    }

    static double window_cov(
            const std::vector<uint64_t> & hashes,
            const std::unordered_map<uint64_t, uint32_t> & counts,
            size_t n, size_t window) {
        if (n < window || hashes.empty()) {
            return 0.0;
        }
        const size_t passages = hashes.size();
        const size_t wstart = n - window;
        std::vector<int> delta(window + 1, 0);
        const size_t first = wstart >= kPassageWords - 1 ? wstart - (kPassageWords - 1) : 0;
        for (size_t i = first; i < passages; ++i) {
            const auto it = counts.find(hashes[i]);
            if (it == counts.end() || it->second < kRepeats) {
                continue;
            }
            const size_t b = std::max(i, wstart) - wstart;
            const size_t e = std::min(i + kPassageWords, n) - wstart;
            delta[b]++;
            delta[e]--;
        }
        size_t covered = 0;
        int depth = 0;
        for (size_t i = 0; i < window; ++i) {
            depth += delta[i];
            if (depth > 0) {
                covered++;
            }
        }
        return double(covered) / double(window);
    }

    bool periodic_tail() const {
        if (tail.size() < kTailChars) {
            return false;
        }
        const std::string sub = tail.substr(tail.size() - kTailChars);
        for (size_t p = 1; p <= kTailPeriod && p < sub.size(); ++p) {
            if (sub.compare(p, sub.size() - p, sub, 0, sub.size() - p) == 0) {
                return true;
            }
        }
        return false;
    }

    bool check_loop() {
        if (!rb_loop_guard_enabled() || tokens_seen < next_check) {
            return false;
        }
        next_check = (tokens_seen / kCheckTokens + 1) * kCheckTokens;
        if (periodic_tail()) {
            return true;
        }
        const size_t n = words.size();
        if (n < kShortWindowWords || n < kPassageWords) {
            return false;
        }
        constexpr uint64_t kBase = 0x9E3779B97F4A7C15ULL;
        uint64_t top = 1;
        for (size_t i = 1; i < kPassageWords; ++i) {
            top *= kBase;
        }
        const size_t passages = n - kPassageWords + 1;
        std::vector<uint64_t> hashes(passages);
        uint64_t h = 0;
        for (size_t i = 0; i < kPassageWords; ++i) {
            h = h * kBase + words[i];
        }
        hashes[0] = h;
        for (size_t i = 1; i < passages; ++i) {
            h = (h - words[i - 1] * top) * kBase + words[i + kPassageWords - 1];
            hashes[i] = h;
        }
        std::unordered_map<uint64_t, uint32_t> counts;
        counts.reserve(passages);
        for (uint64_t x : hashes) {
            counts[x]++;
        }
        if (window_cov(hashes, counts, n, kLongWindowWords) >= kLongCoverage) {
            return true;
        }
        if (tokens_seen <= kShortMaxTokens &&
            window_cov(hashes, counts, n, kShortWindowWords) >= kShortCoverage) {
            if (++short_hits >= kShortConfirm) {
                return true;
            }
        } else {
            short_hits = 0;
        }
        return false;
    }
};

} // namespace

struct token_matcher {
    std::vector<llama_tokens> seqs;
    common_aho_corasick ac;
    size_t state = 0;

    token_matcher(const std::vector<llama_tokens> & seqs) : seqs(collect(seqs)), ac(build_trie(this->seqs)) {}

    static std::vector<llama_tokens> collect(const std::vector<llama_tokens> & seqs) {
        std::vector<llama_tokens> res;
        for (const auto & seq : seqs) {
            if (!seq.empty() && std::find(res.begin(), res.end(), seq) == res.end()) {
                res.push_back(seq);
            }
        }
        return res;
    }

    static common_trie build_trie(const std::vector<llama_tokens> & seqs) {
        common_trie t;
        for (const auto & seq : seqs) {
            t.insert(std::vector<uint32_t>(seq.begin(), seq.end()));
        }
        return t;
    }

    // returns the index into seqs of the longest sequence ending at this token, or -1
    int32_t advance(llama_token token) {
        state = ac.next(state, (uint32_t) token);
        const int32_t p = ac.match_pattern(state);
        if (p >= 0) {
            state = 0;
        }
        return p;
    }

    void reset() { state = 0; }
};

struct common_reasoning_budget_ctx {
    const llama_vocab * vocab;

    token_matcher start_matcher;
    token_matcher end_matcher;
    llama_tokens forced_tokens;

    int32_t budget;           // maximum tokens in reasoning block
    int32_t remaining;        // tokens remaining in budget

    common_reasoning_budget_state state;

    // for forcing
    size_t force_pos;         // next position in forced_tokens to force

    int32_t end_match;        // index into end_matcher.seqs of the sequence that transitioned to DONE, -1 if none

    // anti-loop state: once forced closed (budget or loop detected), lock out re-opening
    // thinking blocks (prevents Inkling 2 from spawning back-to-back <|content_thinking|> loops).
    bool locked;
    reasoning_loop_tracker loop_tracker;
};

static const char * common_reasoning_budget_name(const struct llama_sampler * /*smpl*/) {
    return "reasoning-budget";
}

static void common_reasoning_budget_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    switch (ctx->state) {
        case REASONING_BUDGET_IDLE:
        {
            if (ctx->start_matcher.advance(token) >= 0) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                COM_TRC("activated, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    COM_TRC("%s", "budget=0, forcing immediately\n");
                }
            }
            break;
        }
        case REASONING_BUDGET_COUNTING:
        case REASONING_BUDGET_WAITING_UTF8:
        {
            const int32_t match = ctx->end_matcher.advance(token);
            if (match >= 0) {
                ctx->state = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                COM_TRC("%s", "deactivated (natural end)\n");
                break;
            }

            bool utf8_complete = true;
            std::string piece;
            if (ctx->vocab != nullptr) {
                piece = common_token_to_piece(ctx->vocab, token, false);
                utf8_complete = common_utf8_is_complete(piece);
            }

            if (ctx->state == REASONING_BUDGET_WAITING_UTF8) {
                if (utf8_complete) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    ctx->end_matcher.reset();
                    COM_TRC("%s", "UTF-8 complete, now forcing end sequence\n");
                }
            } else if (ctx->state == REASONING_BUDGET_COUNTING) {
                ctx->remaining--;
                bool loop_detected = false;
                if (!piece.empty()) {
                    ctx->loop_tracker.feed(piece);
                    loop_detected = ctx->loop_tracker.check_loop();
                }
                if (ctx->remaining <= 0 || loop_detected) {
                    if (loop_detected) {
                        LOG_WRN("%s: reasoning loop detected after %d tokens — forcing conclusion\n",
                                __func__, ctx->loop_tracker.tokens_seen);
                    }
                    if (utf8_complete) {
                        ctx->state = REASONING_BUDGET_FORCING;
                        ctx->force_pos = 0;
                        ctx->end_matcher.reset();
                        COM_TRC("%s", "budget/loop triggered, forcing end sequence\n");
                    } else {
                        ctx->state = REASONING_BUDGET_WAITING_UTF8;
                        ctx->end_matcher.reset();
                        COM_TRC("%s", "budget/loop triggered, waiting for UTF-8 completion\n");
                    }
                }
            }
            break;
        }
        case REASONING_BUDGET_FORCING:
        {
            // track the end sequence within forced_tokens so it is also reported on DONE
            const int32_t match = ctx->end_matcher.advance(token);
            ctx->force_pos++;
            // an end tag that is already in the prompt (e.g. "<think></think>") also ends forcing
            if (match >= 0 || ctx->force_pos >= ctx->forced_tokens.size()) {
                ctx->state = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                ctx->locked = true;
                ctx->loop_tracker.reset();
                COM_TRC("%s", "forced sequence complete, locking out thinking re-entry\n");
            }
            break;
        }
        case REASONING_BUDGET_DONE:
            // Do NOT re-arm after a forced budget/loop close (prevents infinite consecutive
            // <|content_thinking|> blocks on Inkling 2). Natural closes may open a new block while
            // retaining loop_tracker word history across blocks so alternating loops stay caught.
            if (!ctx->locked && ctx->start_matcher.advance(token) >= 0) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                ctx->end_matcher.reset();
                ctx->end_match = -1;
                COM_TRC("re-activated on new start tag, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    COM_TRC("%s", "budget=0, forcing immediately\n");
                }
            }
            break;
    }
}

static void common_reasoning_budget_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    if (ctx->locked && ctx->state == REASONING_BUDGET_DONE) {
        // Suppress single-token thinking start markers (e.g. Inkling's <|content_thinking|>) so
        // the model immediately emits a final answer or tool call after forced conclusion.
        for (const auto & seq : ctx->start_matcher.seqs) {
            if (seq.size() == 1) {
                const llama_token ban = seq[0];
                for (size_t i = 0; i < cur_p->size; i++) {
                    if (cur_p->data[i].id == ban) {
                        cur_p->data[i].logit = -INFINITY;
                        break;
                    }
                }
            }
        }
        return;
    }

    if (ctx->state != REASONING_BUDGET_FORCING) {
        // passthrough — don't modify logits
        return;
    }

    if (ctx->force_pos >= ctx->forced_tokens.size()) {
        return;
    }

    const llama_token forced = ctx->forced_tokens[ctx->force_pos];

    // set all logits to -inf except the forced token
    for (size_t i = 0; i < cur_p->size; i++) {
        if (cur_p->data[i].id != forced) {
            cur_p->data[i].logit = -INFINITY;
        }
    }
}

static void common_reasoning_budget_reset(struct llama_sampler * smpl) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;
    ctx->state = REASONING_BUDGET_IDLE;
    ctx->remaining = ctx->budget;
    ctx->start_matcher.reset();
    ctx->end_matcher.reset();
    ctx->force_pos = 0;
    ctx->end_match = -1;
    ctx->locked = false;
    ctx->loop_tracker.reset();
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab * vocab, const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs, const llama_tokens & forced_tokens,
        int32_t budget, common_reasoning_budget_state initial_state);

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl);

static void common_reasoning_budget_free(struct llama_sampler * smpl) {
    delete (common_reasoning_budget_ctx *) smpl->ctx;
}

static struct llama_sampler_i common_reasoning_budget_i = {
    /* .name              = */ common_reasoning_budget_name,
    /* .accept            = */ common_reasoning_budget_accept,
    /* .apply             = */ common_reasoning_budget_apply,
    /* .reset             = */ common_reasoning_budget_reset,
    /* .clone             = */ common_reasoning_budget_clone,
    /* .free              = */ common_reasoning_budget_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ nullptr,
};

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx(*ctx)
    );
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state) {
    // promote COUNTING with budget <= 0 to FORCING
    if (initial_state == REASONING_BUDGET_COUNTING && budget <= 0) {
        initial_state = REASONING_BUDGET_FORCING;
    }

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx {
            /* .vocab         = */ vocab,
            /* .start_matcher = */ token_matcher(start_seqs),
            /* .end_matcher   = */ token_matcher(end_seqs),
            /* .forced_tokens = */ forced_tokens,
            /* .budget        = */ budget,
            /* .remaining     = */ budget,
            /* .state         = */ initial_state,
            /* .force_pos     = */ 0,
            /* .end_match     = */ -1,
            /* .locked        = */ false,
            /* .loop_tracker  = */ {},
        }
    );
}

struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state) {
    return common_reasoning_budget_init_state(vocab, start_seqs, end_seqs, forced_tokens, budget, initial_state);
}

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl) {
    if (!smpl) {
        return REASONING_BUDGET_IDLE;
    }
    return ((const common_reasoning_budget_ctx *)smpl->ctx)->state;
}

const llama_tokens * common_reasoning_budget_get_end_match(const struct llama_sampler * smpl) {
    if (!smpl) {
        return nullptr;
    }

    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;
    if (ctx->end_match < 0) {
        return nullptr;
    }

    return &ctx->end_matcher.seqs[ctx->end_match];
}

bool common_reasoning_budget_force(struct llama_sampler * smpl) {
    if (!smpl) {
        return false;
    }

    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    // only a sampler that is actively counting down the budget may be forced;
    // any other state (idle, already forcing/waiting, or done) is left untouched
    if (ctx->state != REASONING_BUDGET_COUNTING) {
        return false;
    }

    ctx->state = REASONING_BUDGET_FORCING;
    ctx->force_pos = 0;
    ctx->end_matcher.reset();
    COM_TRC("%s", "forced into forcing state (manual transition)\n");

    return true;
}
