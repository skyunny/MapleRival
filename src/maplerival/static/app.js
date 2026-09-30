const colors = ["#baff4a", "#a98bff", "#ff9c46", "#4adfff"];
const $ = (selector) => document.querySelector(selector);
const fmt = new Intl.NumberFormat("ko-KR");
const trillion = (value) => `${(Number(value) / 1e12).toFixed(2)}조`;
const shortDate = (value) => value === "NOW" ? "NOW" : value.slice(5).replace("-", ".");
const owner = new URLSearchParams(window.location.search).get("owner")?.trim();
let pageData = null;

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
}
function eventNotice(notification) { if(!notification?.configured)return ""; return notification.sent?" Discord에도 알렸습니다.":" 목록은 반영됐지만 Discord 메시지 발송에 실패했습니다."; }
function svgElement(tag, attrs = {}) { const node=document.createElementNS("http://www.w3.org/2000/svg",tag); Object.entries(attrs).forEach(([key,value])=>node.setAttribute(key,value)); return node; }
function addText(svg,x,y,text,anchor="start") { const node=svgElement("text",{x,y,"text-anchor":anchor,class:"axis-text"}); node.textContent=text; svg.appendChild(node); }
function renderEmpty(container) { container.innerHTML='<div class="empty">비교할 라이벌을 등록하면 그래프가 완성됩니다.</div>'; }

function nonOverlappingTicks(ticks,y,minGap=18) {
  if(ticks.length<=2)return ticks;
  const ordered=[...ticks].sort((a,b)=>y(Number(a.value))-y(Number(b.value))),selected=[];
  ordered.forEach((tick)=>{const yy=y(Number(tick.value));if(!selected.length||yy-selected[selected.length-1].y>=minGap)selected.push({tick,y:yy});});
  const bottom=ordered[ordered.length-1],bottomY=y(Number(bottom.value));
  if(selected[selected.length-1]?.tick!==bottom){while(selected.length>1&&bottomY-selected[selected.length-1].y<minGap)selected.pop();selected.push({tick:bottom,y:bottomY});}
  return selected.map(({tick})=>tick);
}

function chartFrame(container, values, labels, includeZero = false, customScale = null) {
  const width=1050,height=320,pad={top:20,right:18,bottom:38,left:82};
  const svg=svgElement("svg",{viewBox:`0 0 ${width} ${height}`,role:"img"});
  const flat=values.flat().filter(Number.isFinite),rawMax=Math.max(...flat,1),rawMin=Math.min(...flat,0);
  const min=customScale?0:(includeZero?Math.min(rawMin,0):rawMin),max=customScale?Number(customScale.maxValue):(includeZero?Math.max(rawMax,0):rawMax),spread=Math.max(max-min,max*.02,1),low=customScale?0:min-spread*.12,high=customScale?max:max+spread*.12;
  const innerW=width-pad.left-pad.right,innerH=height-pad.top-pad.bottom,x=(i)=>pad.left+(i/Math.max(labels.length-1,1))*innerW,y=(v)=>pad.top+((high-v)/(high-low))*innerH;
  if(customScale){nonOverlappingTicks(customScale.ticks,y).forEach((tick)=>{const yy=y(Number(tick.value));svg.appendChild(svgElement("line",{x1:pad.left,y1:yy,x2:width-pad.right,y2:yy,class:"grid-line"}));addText(svg,pad.left-10,yy+4,tick.label,"end");});}
  else{for(let i=0;i<5;i+=1){const yy=pad.top+(innerH/4)*i;svg.appendChild(svgElement("line",{x1:pad.left,y1:yy,x2:width-pad.right,y2:yy,class:"grid-line"}));addText(svg,pad.left-10,yy+4,trillion(high-((high-low)/4)*i),"end");}}
  labels.forEach((label,index)=>{if(index%2===0||index===labels.length-1)addText(svg,x(index),height-12,shortDate(label),"middle");});
  container.replaceChildren(svg); return {svg,x,y,pad,innerW};
}

function renderLineChart(characters, experienceScale) {
  const container=$("#lineChart"); if(!characters.length)return renderEmpty(container);
  const labels=[...new Set(characters.flatMap((character)=>character.history.map((point)=>point.date)))].sort();labels.push("NOW");
  const plotted=characters.map((character)=>{const byDate=new Map(character.history.map((point)=>[point.date,point]));return labels.map((label)=>label==="NOW"?{value:Number(character.currentChartExp),level:character.level,rate:character.currentProgressRate}:byDate.has(label)?{value:Number(byDate.get(label).chartExp),level:byDate.get(label).level,rate:byDate.get(label).progressRate}:null);});
  const frame=chartFrame(container,plotted.map((points)=>points.map((point)=>point?.value)),labels,false,experienceScale);
  plotted.forEach((points,seriesIndex)=>{const character=characters[seriesIndex],available=points.map((point,index)=>({point,index})).filter(({point})=>point!==null),path=available.map(({point,index},pathIndex)=>`${pathIndex?"L":"M"}${frame.x(index)},${frame.y(point.value)}`).join(" ");frame.svg.appendChild(svgElement("path",{d:path,stroke:colors[seriesIndex],class:"line-path"}));available.forEach(({point:meta,index})=>{const point=svgElement("circle",{cx:frame.x(index),cy:frame.y(meta.value),r:4,fill:colors[seriesIndex],class:"point"}),title=svgElement("title");title.textContent=`${character.name} · ${labels[index]} · Lv.${meta.level} · ${Number(meta.rate).toFixed(3)}%`;point.appendChild(title);frame.svg.appendChild(point);});});
}

function renderBarChart(characters) {
  const container=$("#barChart"); if(!characters.length)return renderEmpty(container);
  const labels=characters[0].history.map((point)=>point.date),series=characters.map((character)=>character.history.map((point)=>Number(point.dailyGain||0))),frame=chartFrame(container,series,labels,true),groupWidth=frame.innerW/labels.length,barWidth=Math.min(18,groupWidth*.72/characters.length),zeroY=frame.y(0);
  series.forEach((values,seriesIndex)=>values.forEach((value,index)=>{const offset=(seriesIndex-(characters.length-1)/2)*barWidth,yy=frame.y(value),bar=svgElement("rect",{x:frame.pad.left+index*groupWidth+groupWidth/2+offset-barWidth/2,y:Math.min(yy,zeroY),width:Math.max(barWidth-2,2),height:Math.max(Math.abs(zeroY-yy),1),rx:3,fill:colors[seriesIndex],class:"bar"}),title=svgElement("title");title.textContent=`${characters[seriesIndex].name} · ${labels[index]} · ${fmt.format(value)} EXP`;bar.appendChild(title);frame.svg.appendChild(bar);}));
}

function characterCard(character, index, isOwner) {
  const rate=character.expRate==null?null:Number(character.expRate),name=escapeHtml(character.name);
  return `<article class="character-card" style="--accent:${colors[index]}"><div class="character-top"><div><div class="name-line"><h3 class="character-name">${name}</h3>${isOwner?'<span class="owner-badge">ME</span>':""}</div><span class="character-meta">${escapeHtml(character.world||"—")} · ${escapeHtml(character.className||"—")}</span></div><div class="card-actions"><span class="rank">#${character.ranking?fmt.format(character.ranking):"—"}</span>${isOwner?"":`<button class="delete-rival" data-rival="${name}" type="button" aria-label="${name} 삭제">×</button>`}</div></div><div class="level-progress"><div class="level-row"><strong>Lv.${character.level}</strong><span>${rate==null?"—":rate.toFixed(3)+"%"}</span></div><div class="progress-track"><i style="width:${rate==null?0:Math.min(100,Math.max(0,rate))}%"></i></div></div><div class="metric metric-secondary"><span class="metric-label">CURRENT EXP · NOW</span><strong class="metric-value">${fmt.format(character.latestExp)}</strong></div></article>`;
}
function addCard() { return `<button class="add-rival-card" id="openRivalDialog" type="button"><span class="plus-ring">+</span><strong>라이벌 캐릭터를 등록하세요.</strong><small>최대 3명까지 비교할 수 있어요</small></button>`; }

function render(result) {
  pageData=result; const data=result.dashboard;
  $("#ownerHeading").textContent=result.owner;
  $("#leader").textContent=data.leader||result.owner;
  $("#gap").textContent=result.rivals.length?`${data.leader} 선두 · 상위 두 캐릭터 ${data.gap?trillion(data.gap):"레벨 차이"}`:"라이벌을 등록해 주세요";
  $("#latestDate").textContent=`현재값 확인 ${data.currentCheckedAt?new Date(data.currentCheckedAt).toLocaleString("ko-KR"):"—"}`;
  const cards=data.characters.map((character,index)=>characterCard(character,index,character.name===result.owner));
  if(result.rivals.length<result.maximumRivals)cards.push(addCard());
  $("#characterCards").innerHTML=cards.join("");
  $("#lineLegend").innerHTML=data.characters.map((character,index)=>`<span style="--dot:${colors[index]}">${escapeHtml(character.name)}</span>`).join("");
  renderLineChart(data.characters,data.experienceScale); renderBarChart(data.characters); bindCardActions();
}

function bindCardActions() {
  $("#openRivalDialog")?.addEventListener("click",()=>{$("#dialogMessage").textContent="";$("#rivalName").value="";$("#rivalDialog").showModal();$("#rivalName").focus();});
  document.querySelectorAll(".delete-rival").forEach((button)=>button.addEventListener("click",async()=>{const rival=button.dataset.rival;if(!window.confirm(`${rival}님을 라이벌 목록에서 삭제할까요?`))return;button.disabled=true;try{const response=await fetch(`/api/owners/${encodeURIComponent(owner)}/rivals/${encodeURIComponent(rival)}`,{method:"DELETE"}),result=await response.json();if(!response.ok)throw new Error(result.detail||"삭제하지 못했습니다.");render(result);await loadWebhookStatus();$("#message").className=result.notification?.configured&&!result.notification.sent?"message error":"message";$("#message").textContent=`${rival}님을 라이벌 목록에서 삭제했습니다.${eventNotice(result.notification)}`;}catch(error){$("#message").className="message error";$("#message").textContent=error.message;}}));
}

async function loadDashboard(){if(!owner){window.location.href="/";return;}const response=await fetch(`/api/owners/${encodeURIComponent(owner)}/dashboard`),result=await response.json();if(!response.ok)throw new Error(result.detail||"대시보드를 불러오지 못했습니다.");render(result);}
async function loadWebhookStatus(){const response=await fetch(`/api/owners/${encodeURIComponent(owner)}/webhook`),result=await response.json();if(!response.ok)return;const status=$("#webhookStatus");status.textContent=result.configured?`알림 연결됨 · 라이벌 ${result.rivalStates.length}명`:"알림 미설정";status.classList.toggle("configured",result.configured);}
async function refresh(){const button=$("#refreshButton");button.disabled=true;button.textContent="수집 중…";$("#message").className="message";$("#message").textContent="등록된 캐릭터들의 현재값과 최근 기록을 갱신하고 있습니다.";try{const response=await fetch(`/api/owners/${encodeURIComponent(owner)}/refresh`,{method:"POST"}),result=await response.json();if(!response.ok)throw new Error(result.detail||"데이터 수집에 실패했습니다.");render(result);$("#message").textContent="모든 캐릭터 데이터를 갱신했습니다.";}catch(error){$("#message").className="message error";$("#message").textContent=error.message;}finally{button.disabled=false;button.textContent="데이터 새로고침";}}

$("#refreshButton").addEventListener("click",refresh);
$("#webhookButton").addEventListener("click",()=>{$("#webhookUrl").value="";$("#webhookMessage").textContent="";$("#webhookDialog").showModal();$("#webhookUrl").focus();});
$("#closeDialog").addEventListener("click",()=>$("#rivalDialog").close());
$("#closeWebhookDialog").addEventListener("click",()=>$("#webhookDialog").close());
$("#rivalForm").addEventListener("submit",async(event)=>{event.preventDefault();const name=$("#rivalName").value.trim(),button=$("#addRivalButton");if(!name)return;button.disabled=true;button.textContent="15일 수집 중…";$("#dialogMessage").className="message";$("#dialogMessage").textContent=`${name}님의 데이터를 준비하고 있습니다.`;try{const response=await fetch(`/api/owners/${encodeURIComponent(owner)}/rivals`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name})}),result=await response.json();if(!response.ok)throw new Error(result.detail||"라이벌을 등록하지 못했습니다.");$("#rivalDialog").close();render(result);await loadWebhookStatus();$("#message").className=result.notification?.configured&&!result.notification.sent?"message error":"message";$("#message").textContent=`${name}님을 새로운 라이벌로 등록했습니다.${eventNotice(result.notification)}`;}catch(error){$("#dialogMessage").className="message error";$("#dialogMessage").textContent=error.message;}finally{button.disabled=false;button.textContent="라이벌 추가";}});
$("#webhookForm").addEventListener("submit",async(event)=>{event.preventDefault();const url=$("#webhookUrl").value.trim(),button=$("#saveWebhookButton");if(!url)return;button.disabled=true;button.textContent="현재 상태 저장 중…";$("#webhookMessage").className="message";$("#webhookMessage").textContent="모든 라이벌의 최초 격차 상태를 확인하고 있습니다.";try{const response=await fetch(`/api/owners/${encodeURIComponent(owner)}/webhook`,{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({url})}),result=await response.json();if(!response.ok)throw new Error(result.detail||"웹훅을 저장하지 못했습니다.");$("#webhookDialog").close();await loadWebhookStatus();const sent=result.notification?.sent;$("#message").className=sent?"message":"message error";$("#message").textContent=sent?`Discord 알림을 연결하고 라이벌 ${result.rivalStates.length}명의 최초 상태를 저장했습니다. 시작 메시지를 발송했습니다.`:`웹훅 설정은 저장됐지만 시작 메시지 발송에 실패했습니다.`;}catch(error){$("#webhookMessage").className="message error";$("#webhookMessage").textContent=error.message;}finally{button.disabled=false;button.textContent="웹훅 저장";}});
Promise.all([loadDashboard(),loadWebhookStatus()]).catch((error)=>{$("#message").className="message error";$("#message").textContent=error.message;});
