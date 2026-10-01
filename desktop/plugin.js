/**
 * hermes-council — desktop panel.
 *
 * /council page: the 4 fixed seats (name + duty are static product copy from
 * the backend), each with provider/model dropdowns; optional Judge; Chairman.
 * Autosaves via ctx.rest POST /config. Run history at the bottom.
 *
 * All model calls happen in the gateway (dashboard/plugin_api.py + engine) and
 * borrow the user's existing provider credentials — this half is pure UI.
 */

import {
	Button,
	cn,
	host,
	Loader,
	ROUTES_AREA,
	ScrollArea,
	SIDEBAR_NAV_AREA,
	Switch,
	useMutation,
	useQuery,
	useQueryClient,
} from "@hermes/plugin-sdk";
import { useEffect, useMemo, useState } from "react";
import { jsx, jsxs } from "react/jsx-runtime";

const ID = "hermes-council";
let CTX = null;

function useCouncilConfig() {
	return useQuery({
		queryKey: [ID, "config"],
		queryFn: () => CTX.rest("/config"),
		staleTime: 30_000,
	});
}

function useProviders() {
	return useQuery({
		queryKey: [ID, "models"],
		queryFn: async () => (await CTX.rest("/models")).providers || [],
		staleTime: 60_000,
	});
}

function useRuns() {
	return useQuery({
		queryKey: [ID, "runs"],
		queryFn: async () => (await CTX.rest("/runs?limit=20")).runs || [],
	});
}

function ModelPicker({ slot, onChange, providers }) {
	const models = useMemo(() => {
		const p = (providers || []).find((x) => x.provider === slot.provider);
		return p ? p.models : [];
	}, [providers, slot.provider]);

	const selectCls =
		"rounded-md border border-(--ui-stroke-tertiary) bg-transparent px-2 py-1 text-xs text-(--ui-text-primary) outline-none focus:border-(--ui-accent)";

	return jsxs("div", {
		className: "flex flex-wrap items-center gap-2",
		children: [
			jsx("select", {
				className: selectCls,
				value: slot.provider || "",
				onChange: (e) => onChange({ ...slot, provider: e.target.value, model: "" }),
				children: [
					jsx("option", { value: "", children: "Provider…" }),
					...(providers || []).map((p) =>
						jsx("option", { value: p.provider, children: p.name || p.provider }, p.provider),
					),
				],
			}),
			jsx("select", {
				className: cn(selectCls, "min-w-40"),
				value: slot.model || "",
				disabled: !slot.provider,
				onChange: (e) => onChange({ ...slot, model: e.target.value }),
				children: [
					jsx("option", { value: "", children: "Model…" }),
					...models.map((m) => jsx("option", { value: m, children: m }, m)),
				],
			}),
		],
	});
}

function CouncilPage() {
	const qc = useQueryClient();
	const cfgQ = useCouncilConfig();
	const provQ = useProviders();
	const runsQ = useRuns();
	const [draft, setDraft] = useState(null);

	// Seed the draft from the loaded config once.
	useEffect(() => {
		if (cfgQ.data?.preset && !draft) setDraft(cfgQ.data.preset);
	}, [cfgQ.data, draft]);

	const save = useMutation({
		mutationFn: (preset) =>
			CTX.rest("/config", {
				method: "POST",
				body: { presets: { default: preset }, default_preset: "default" },
			}),
		onSuccess: (res) => {
			if (res?.error) host.notifyError?.(res.error);
			else qc.invalidateQueries({ queryKey: [ID, "config"] });
		},
		onError: (e) => host.notifyError?.(String(e?.message || e)),
	});

	// Debounced autosave.
	useEffect(() => {
		if (!draft) return;
		const t = setTimeout(() => save.mutate(draft), 600);
		return () => clearTimeout(t);
	}, [draft]); // eslint-disable-line react-hooks/exhaustive-deps

	const setSeat = (i, slot) =>
		setDraft((prev) => ({ ...prev, members: prev.members.map((s, j) => (j === i ? { ...s, ...slot } : s)) }));

	if (cfgQ.isLoading || provQ.isLoading) return jsx(Loader, {});
	if (!draft) return jsx("div", { className: "p-5 text-xs text-(--ui-text-tertiary)", children: "Loading council…" });

	const seats = cfgQ.data?.seats || [];

	return jsx(ScrollArea, {
		className: "h-full",
		children: jsxs("div", {
			className: "mx-auto max-w-2xl space-y-4 p-5",
			children: [
				jsxs("div", {
					children: [
						jsx("h1", { className: "text-lg font-semibold text-(--ui-text-primary)", children: "LLM Council" }),
						jsx("p", {
							className: "mt-0.5 text-xs text-(--ui-text-tertiary)",
							children:
								"Four fixed seats answer in parallel, anonymously review and rank each other, then the Chairman synthesizes the final answer. Pick a model per seat — your existing provider credentials are used; no API keys here.",
						}),
					],
				}),

				...seats.map((seat, i) =>
					jsx(
						"div",
						{
							className: "rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-elevated) p-3",
							children: jsxs("div", {
								children: [
									jsxs("div", {
										className: "flex items-baseline justify-between gap-2",
										children: [
											jsx("span", { className: "text-xs font-semibold text-(--ui-text-primary)", children: seat.name }),
											jsx("span", {
												className: "font-mono text-[0.625rem] text-(--ui-text-quaternary)",
												children:
													draft.members[i]?.provider && draft.members[i]?.model
														? `${draft.members[i].provider} · ${draft.members[i].model}`
														: "no model selected",
											}),
										],
									}),
									jsx("p", {
										className: "mt-0.5 text-[0.6875rem] leading-snug text-(--ui-text-tertiary)",
										children: seat.role,
									}),
									jsx("div", {
										className: "mt-2",
										children: jsx(ModelPicker, {
											slot: draft.members[i] || { provider: "", model: "" },
											providers: provQ.data,
											onChange: (slot) => setSeat(i, slot),
										}),
									}),
								],
							}),
						},
						seat.name,
					),
				),

				// Judge
				jsxs("div", {
					className: "rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-elevated) p-3",
					children: [
						jsxs("div", {
							className: "flex items-baseline justify-between gap-2",
							children: [
								jsx("span", { className: "text-xs font-semibold text-(--ui-text-primary)", children: "Judge" }),
								jsx("span", { className: "text-[0.625rem] text-(--ui-text-quaternary)", children: "optional" }),
							],
						}),
						jsx("p", {
							className: "mt-0.5 text-[0.6875rem] text-(--ui-text-tertiary)",
							children: "Empty: the seats review each other anonymously. Set a judge to rank in one call instead.",
						}),
						jsx("div", {
							className: "mt-2",
							children: jsx(ModelPicker, {
								slot: draft.judge || { provider: "", model: "" },
								providers: provQ.data,
								onChange: (slot) => setDraft((prev) => ({ ...prev, judge: { ...prev.judge, ...slot } })),
							}),
						}),
					],
				}),

				// Chairman
				jsxs("div", {
					className: "rounded-lg border border-(--ui-accent)/40 bg-(--ui-accent)/5 p-3",
					children: [
						jsxs("div", {
							className: "flex items-baseline justify-between gap-2",
							children: [
								jsx("span", { className: "text-xs font-semibold text-(--ui-text-primary)", children: "Chairman" }),
								jsx("span", { className: "text-[0.625rem] text-(--ui-text-quaternary)", children: "acting model · billed for the run" }),
							],
						}),
						jsx("div", {
							className: "mt-2",
							children: jsx(ModelPicker, {
								slot: draft.chairman || { provider: "", model: "" },
								providers: provQ.data,
								onChange: (slot) => setDraft((prev) => ({ ...prev, chairman: { ...prev.chairman, ...slot } })),
							}),
						}),
					],
				}),

				// Lite
				jsxs("label", {
					className: "flex items-center gap-2 text-xs text-(--ui-text-secondary)",
					children: [
						jsx(Switch, {
							checked: !!draft.lite,
							onCheckedChange: (v) => setDraft((prev) => ({ ...prev, lite: v === true })),
						}),
						"Lite — skip peer review (N+1 calls instead of 2N+1)",
					],
				}),

				jsx(Button, {
					onClick: () => save.mutate(draft),
					disabled: save.isPending,
					children: save.isPending ? "Saving…" : "Save council",
				}),

				runsQ.data?.length
					? jsxs("div", {
							children: [
								jsx("div", {
									className: "mb-2 text-[0.6875rem] font-semibold uppercase tracking-wide text-(--ui-text-tertiary)",
									children: "Recent deliberations",
								}),
								jsx("div", {
									className: "space-y-1",
									children: runsQ.data.map((r) =>
										jsxs("div", {
											className: "rounded-md border border-(--ui-stroke-tertiary) px-2.5 py-1.5 text-xs",
											children: [
												jsx("div", { className: "truncate text-(--ui-text-primary)", children: r.query }),
												jsx("div", { className: "text-[0.625rem] text-(--ui-text-quaternary)", children: r.run_id }),
											],
										}, r.run_id),
									),
								}),
							],
						})
					: null,
			],
		}),
	});
}

export default {
	id: ID,
	name: "LLM Council",
	defaultEnabled: true,
	register(ctx) {
		CTX = ctx;
		ctx.register({
			id: "page",
			area: ROUTES_AREA,
			data: { path: "/council" },
			render: () => jsx(CouncilPage, {}),
		});
		ctx.register({
			id: "nav",
			area: SIDEBAR_NAV_AREA,
			order: 85,
			data: { codicon: "organization", label: "Council", path: "/council" },
		});
	},
};
