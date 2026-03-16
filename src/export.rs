/// JSON export for solver results.

use crate::card::{card_to_string, combo_from_index};
use crate::cfr::SubgameSolver;
use crate::game::Action;
use crate::strategy::PreflopChart;
use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// Subgame solve result
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct SolveResult {
    pub config: SolveConfig,
    pub strategy: Vec<NodeStrategy>,
    pub iterations: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exploitability_pct: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oop_ev: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ip_ev: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SolveConfig {
    pub board: String,
    pub pot: i32,
    pub stacks: [i32; 2],
    pub street: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NodeStrategy {
    pub node: String,           // action sequence description
    pub player: String,         // "OOP" or "IP"
    pub combos: Vec<ComboStrategy>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComboStrategy {
    pub hand: String,           // e.g., "AsKh"
    pub actions: Vec<ActionWeight>,
    pub ev: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ActionWeight {
    pub action: String,
    pub weight: f32,
}

impl SolveResult {
    /// Export subgame solver results.
    pub fn from_solver(solver: &SubgameSolver) -> Self {
        let mut board_cards: Vec<_> = solver.config.board.iter().collect();
        board_cards.sort_unstable_by(|a, b| b.cmp(a)); // high cards first
        let board_str: String = board_cards.iter()
            .map(|c| card_to_string(*c))
            .collect::<Vec<_>>()
            .join("");

        let street_str = match solver.config.street {
            crate::game::Street::Preflop => "preflop",
            crate::game::Street::Flop => "flop",
            crate::game::Street::Turn => "turn",
            crate::game::Street::River => "river",
        };

        let config = SolveConfig {
            board: board_str,
            pot: solver.config.pot,
            stacks: solver.config.stacks,
            street: street_str.to_string(),
        };

        // Compute EVs for both players
        let oop_ev = solver.compute_ev(crate::game::OOP);
        let ip_ev = solver.compute_ev(crate::game::IP);

        // Extract root node strategy
        let mut strategies = Vec::new();
        if let Some(combos) = solver.get_strategy(&[]) {
            let node_strat = build_node_strategy("root", "OOP", &combos, Some(&oop_ev));
            strategies.push(node_strat);
        }

        // Walk through nodes to find all decision points
        for action_seq in solver.iter_decision_nodes() {
            if action_seq.is_empty() {
                continue; // already handled root
            }
            if let Some(combos) = solver.get_strategy(action_seq) {
                let node_desc = action_seq.iter()
                    .map(|a| format!("{}", a))
                    .collect::<Vec<_>>()
                    .join(" → ");
                let (player, ev_ref) = match solver.get_player(action_seq) {
                    Some(crate::game::OOP) => ("OOP", &oop_ev),
                    Some(crate::game::IP) => ("IP", &ip_ev),
                    _ => if action_seq.len() % 2 == 0 {
                        ("OOP", &oop_ev)
                    } else {
                        ("IP", &ip_ev)
                    },
                };
                let node_strat = build_node_strategy(&node_desc, player, &combos, Some(ev_ref));
                strategies.push(node_strat);
            }
        }

        let (oop_overall, ip_overall) = solver.overall_ev();

        SolveResult {
            config,
            strategy: strategies,
            iterations: solver.config.iterations,
            exploitability_pct: None,
            oop_ev: Some(oop_overall),
            ip_ev: Some(ip_overall),
        }
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Export as a self-contained HTML strategy viewer.
    pub fn to_html(&self) -> String {
        let json_data = serde_json::to_string(self).unwrap_or_default();
        format!(r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DCFR Solver - {board} ({street})</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; }}
.header {{ background: #16213e; padding: 16px 24px; border-bottom: 2px solid #0f3460; }}
.header h1 {{ font-size: 20px; color: #e94560; }}
.header .info {{ font-size: 13px; color: #888; margin-top: 4px; }}
.container {{ display: flex; height: calc(100vh - 72px); }}
.sidebar {{ width: 280px; background: #16213e; overflow-y: auto; border-right: 1px solid #0f3460; flex-shrink: 0; }}
.sidebar .node {{ padding: 10px 16px; cursor: pointer; border-bottom: 1px solid #0f3460; font-size: 13px; }}
.sidebar .node:hover {{ background: #1a2a4e; }}
.sidebar .node.active {{ background: #0f3460; color: #e94560; }}
.sidebar .node .player {{ font-size: 11px; color: #888; }}
.main {{ flex: 1; overflow-y: auto; padding: 20px; }}
.strategy-grid {{ display: grid; grid-template-columns: repeat(13, 1fr); gap: 2px; max-width: 700px; margin: 0 auto 20px; }}
.cell {{ aspect-ratio: 1; display: flex; flex-direction: column; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 600; border-radius: 3px; cursor: pointer; position: relative; overflow: hidden; }}
.cell .label {{ position: relative; z-index: 1; text-shadow: 0 1px 2px rgba(0,0,0,0.8); }}
.cell .bar {{ position: absolute; bottom: 0; left: 0; right: 0; }}
.cell:hover {{ outline: 2px solid #e94560; z-index: 10; }}
table {{ border-collapse: collapse; width: 100%; max-width: 900px; margin: 20px auto; font-size: 13px; }}
th {{ background: #16213e; padding: 8px 12px; text-align: left; border-bottom: 2px solid #0f3460; }}
td {{ padding: 6px 12px; border-bottom: 1px solid #222; }}
tr:hover {{ background: #1a2a4e; }}
.action-bar {{ display: inline-block; height: 14px; border-radius: 2px; margin-right: 2px; vertical-align: middle; }}
.legend {{ display: flex; gap: 16px; margin: 16px auto; max-width: 700px; flex-wrap: wrap; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 12px; }}
.legend-dot {{ width: 14px; height: 14px; border-radius: 3px; }}
.summary {{ max-width: 700px; margin: 0 auto 20px; background: #16213e; padding: 16px; border-radius: 8px; }}
.summary h3 {{ margin-bottom: 8px; color: #e94560; }}
.tabs {{ display: flex; gap: 8px; max-width: 700px; margin: 0 auto 16px; }}
.tab {{ padding: 6px 16px; border-radius: 4px; cursor: pointer; background: #16213e; font-size: 13px; }}
.tab.active {{ background: #e94560; color: white; }}
#detail {{ max-width: 900px; margin: 0 auto; }}
</style>
</head>
<body>
<div class="header">
  <h1>DCFR Solver</h1>
  <div class="info" id="header-info"></div>
</div>
<div class="container">
  <div class="sidebar" id="sidebar"></div>
  <div class="main" id="main"></div>
</div>
<script>
const DATA = {json};
const ACTION_COLORS = {{
  'fold': '#666', 'check': '#4CAF50', 'call': '#4CAF50',
  'bet 33%': '#2196F3', 'bet 67%': '#FF9800', 'bet 75%': '#FF9800',
  'bet 100%': '#e94560', 'bet 125%': '#e94560', 'bet 150%': '#9C27B0',
  'bet 200%': '#9C27B0', 'allin': '#e94560',
}};
function actionColor(name) {{
  const n = name.toLowerCase();
  for (const [k,v] of Object.entries(ACTION_COLORS)) {{ if (n.includes(k) || n === k) return v; }}
  if (n.includes('bet')) return '#FF9800';
  if (n.includes('raise')) return '#FF5722';
  return '#888';
}}
const RANKS = ['A','K','Q','J','T','9','8','7','6','5','4','3','2'];
function handCategory(h) {{
  const r1 = h[0], s1 = h[1], r2 = h[2], s2 = h[3];
  const ri1 = RANKS.indexOf(r1), ri2 = RANKS.indexOf(r2);
  if (r1 === r2) return r1 + r2;
  const suited = s1 === s2;
  const [hi, lo] = ri1 < ri2 ? [r1, r2] : [r2, r1];
  return hi + lo + (suited ? 's' : 'o');
}}
function renderSidebar() {{
  const sb = document.getElementById('sidebar');
  DATA.strategy.forEach((ns, i) => {{
    const d = document.createElement('div');
    d.className = 'node' + (i === 0 ? ' active' : '');
    d.innerHTML = `<div>${{ns.node || 'root'}}</div><div class="player">${{ns.player}} (${{ns.combos.length}} combos)</div>`;
    d.onclick = () => {{ document.querySelectorAll('.node').forEach(n => n.classList.remove('active')); d.classList.add('active'); renderNode(i); }};
    sb.appendChild(d);
  }});
}}
function renderNode(idx) {{
  const ns = DATA.strategy[idx];
  const main = document.getElementById('main');
  // Collect unique actions
  const actionSet = new Set();
  ns.combos.forEach(c => c.actions.forEach(a => actionSet.add(a.action)));
  const actions = Array.from(actionSet);
  // Group by hand category
  const groups = {{}};
  ns.combos.forEach(c => {{
    const cat = handCategory(c.hand);
    if (!groups[cat]) groups[cat] = [];
    groups[cat].push(c);
  }});
  // Compute category averages
  const catAvg = {{}};
  for (const [cat, combos] of Object.entries(groups)) {{
    const avg = {{}};
    actions.forEach(a => avg[a] = 0);
    let total = 0;
    combos.forEach(c => {{
      total++;
      c.actions.forEach(a => {{ avg[a.action] = (avg[a.action] || 0) + a.weight; }});
    }});
    for (const a of actions) avg[a] /= total;
    catAvg[cat] = avg;
  }}
  // Aggregate
  let totalW = 0; const aggW = {{}};
  actions.forEach(a => aggW[a] = 0);
  ns.combos.forEach(c => {{ totalW++; c.actions.forEach(a => {{ aggW[a.action] += a.weight; }}); }});
  // Legend
  let legend = '<div class="legend">';
  actions.forEach(a => {{
    const pct = (aggW[a] / totalW * 100).toFixed(1);
    legend += `<div class="legend-item"><div class="legend-dot" style="background:${{actionColor(a)}}"></div>${{a}} ${{pct}}%</div>`;
  }});
  legend += '</div>';
  // Summary
  let summary = `<div class="summary"><h3>${{ns.node || 'Root'}} (${{ns.player}})</h3>`;
  summary += `<div>${{ns.combos.length}} combos | Actions: ${{actions.join(', ')}}</div></div>`;
  // 13x13 grid
  let grid = '<div class="strategy-grid">';
  for (let r = 0; r < 13; r++) {{
    for (let c = 0; c < 13; c++) {{
      let cat;
      if (r === c) cat = RANKS[r] + RANKS[c];
      else if (r < c) cat = RANKS[r] + RANKS[c] + 's';
      else cat = RANKS[c] + RANKS[r] + 'o';
      const avg = catAvg[cat];
      let bg = '#222';
      let bars = '';
      if (avg) {{
        let cumH = 0;
        actions.forEach(a => {{
          const h = avg[a] * 100;
          if (h > 0.5) {{
            bars += `<div class="bar" style="background:${{actionColor(a)}};height:${{h}}%;bottom:${{cumH}}%"></div>`;
            cumH += h;
          }}
        }});
      }}
      grid += `<div class="cell" style="background:#222" data-cat="${{cat}}" onclick="showDetail('${{cat}}')"><div class="label">${{cat}}</div>${{bars}}</div>`;
    }}
  }}
  grid += '</div>';
  // Tabs
  let tabs = '<div class="tabs"><div class="tab active" onclick="showView(\'grid\')">Grid</div><div class="tab" onclick="showView(\'table\')">Table</div></div>';
  // Table
  let table = '<div id="detail" style="display:none"><table><thead><tr><th>Hand</th>';
  actions.forEach(a => {{ table += `<th style="color:${{actionColor(a)}}">${{a}}</th>`; }});
  table += '<th>EV</th></tr></thead><tbody>';
  const sorted = [...ns.combos].sort((a,b) => b.ev - a.ev);
  sorted.forEach(c => {{
    table += `<tr><td><b>${{c.hand}}</b></td>`;
    actions.forEach(a => {{
      const aw = c.actions.find(x => x.action === a);
      const w = aw ? aw.weight : 0;
      const pct = (w * 100).toFixed(0);
      if (w > 0.01) {{
        table += `<td><span class="action-bar" style="background:${{actionColor(a)}};width:${{Math.max(w*60,4)}}px"></span> ${{pct}}%</td>`;
      }} else {{
        table += '<td>-</td>';
      }}
    }});
    table += `<td>${{c.ev.toFixed(1)}}</td></tr>`;
  }});
  table += '</tbody></table></div>';
  main.innerHTML = summary + legend + tabs + '<div id="gridView">' + grid + '</div>' + table;
  window._actions = actions; window._groups = groups; window._ns = ns;
}}
function showView(v) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  if (v === 'grid') {{
    document.getElementById('gridView').style.display = '';
    document.getElementById('detail').style.display = 'none';
    document.querySelectorAll('.tab')[0].classList.add('active');
  }} else {{
    document.getElementById('gridView').style.display = 'none';
    document.getElementById('detail').style.display = '';
    document.querySelectorAll('.tab')[1].classList.add('active');
  }}
}}
function showDetail(cat) {{
  showView('table');
  // Scroll to first matching hand
  const rows = document.querySelectorAll('#detail tbody tr');
  for (const row of rows) {{
    const hand = row.cells[0].textContent;
    if (handCategory(hand) === cat) {{ row.scrollIntoView({{block:'center'}}); row.style.background='#2a3a5e'; setTimeout(()=>row.style.background='',1500); break; }}
  }}
}}
let hdr = `Board: ${{DATA.config.board}} | Street: ${{DATA.config.street}} | Pot: ${{DATA.config.pot}} | Stacks: ${{DATA.config.stacks.join('/')}} | Iterations: ${{DATA.iterations}}`;
if (DATA.exploitability_pct != null) hdr += ` | Expl: ${{DATA.exploitability_pct.toFixed(4)}}%`;
if (DATA.oop_ev != null && DATA.ip_ev != null) hdr += ` | OOP EV: ${{DATA.oop_ev.toFixed(2)}} | IP EV: ${{DATA.ip_ev.toFixed(2)}}`;
document.getElementById('header-info').textContent = hdr;
renderSidebar();
if (DATA.strategy.length > 0) renderNode(0);
</script>
</body>
</html>"##, board = self.config.board, street = self.config.street, json = json_data)
    }
}

fn build_node_strategy(
    node: &str,
    player: &str,
    combos: &[(u16, Vec<(Action, f32)>)],
    ev_array: Option<&[f32; 1326]>,
) -> NodeStrategy {
    let mut combo_strategies = Vec::new();

    for &(idx, ref action_probs) in combos {
        let (c1, c2) = combo_from_index(idx);
        let hand = format!("{}{}", card_to_string(c1), card_to_string(c2));

        let actions: Vec<ActionWeight> = action_probs.iter()
            .filter(|(_, p)| *p > 0.001)
            .map(|(a, p)| ActionWeight {
                action: format!("{}", a),
                weight: *p,
            })
            .collect();

        let ev = ev_array.map_or(0.0, |evs| evs[idx as usize]);

        if !actions.is_empty() {
            combo_strategies.push(ComboStrategy {
                hand,
                actions,
                ev,
            });
        }
    }

    NodeStrategy {
        node: node.to_string(),
        player: player.to_string(),
        combos: combo_strategies,
    }
}

/// Export preflop chart to JSON.
pub fn export_preflop_chart(chart: &PreflopChart) -> String {
    serde_json::to_string_pretty(chart).unwrap_or_default()
}
