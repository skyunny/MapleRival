# Maple Rival

NEXON Open API의 메이플스토리 종합 랭킹 데이터를 매일 저장하고 두 캐릭터의 경험치 추이를 비교하는 포트폴리오용 대시보드입니다.

## 현재 기능

- 꽃게쥬와 탕무스의 최근 15일 랭킹 데이터 수집
- SQLite 스냅샷 저장 및 날짜별 중복 방지
- 경험치 총량 꺾은선그래프
- 일일 획득 경험치 막대그래프
- 현재 선두, 격차, 레벨, 직업, 종합 랭킹 표시
- API `429` 응답 재시도
- 1시간마다 현재 레벨·진행률 확인
- 1%p 경계를 기준으로 한 4단계 경쟁 상태 저장
- 상태가 바뀔 때만 Discord 웹훅 알림 발송
- 알림 성공·실패 이력 저장
- 내 캐릭터별 Discord 웹훅 암호화 저장
- 라이벌별 독립 상태·마지막 확인·알림 이력 관리
- 캐릭터명 검색 전용 랜딩 페이지
- 내 캐릭터별 라이벌 최대 3명 등록·삭제
- 내 캐릭터를 포함한 최대 4명 비교 차트
- 라이벌 등록 즉시 최근 15일 데이터 수집

## 실행

`.env`에 `NEXON_API_KEY`와 `DISCORD_WEBHOOK_URL`을 입력한 뒤 실행합니다.

```powershell
uv sync
uv run uvicorn maplerival.main:app --reload
```

브라우저에서 <http://127.0.0.1:8000>을 엽니다. 최초 실행 시 화면의 **데이터 새로고침** 버튼이 최근 15일 데이터를 수집합니다.

## API

- `GET /api/dashboard`: 저장된 비교 데이터
- `POST /api/refresh`: 최근 15일 수집 및 저장
- `POST /api/characters/select`: 내 캐릭터 검색 및 데이터 준비
- `GET /api/owners/{name}/dashboard`: 내 캐릭터와 등록 라이벌 조회
- `POST /api/owners/{name}/rivals`: 라이벌 등록 및 데이터 수집
- `DELETE /api/owners/{name}/rivals/{rival}`: 라이벌 삭제
- `POST /api/owners/{name}/refresh`: 등록된 전체 캐릭터 갱신
- `GET /api/owners/{name}/webhook`: 웹훅 설정 및 라이벌별 알림 상태
- `PUT /api/owners/{name}/webhook`: 웹훅 등록과 전체 라이벌 최초 상태 저장
- `POST /api/owners/{name}/alerts/check`: 모든 라이벌 상태 즉시 확인
- `GET /health`: 서버 상태
- `GET /api/alerts/status`: 현재 알림 상태와 마지막 확인 시각
- `POST /api/alerts/check`: 즉시 상태 확인 및 필요한 알림 발송
- `GET /docs`: FastAPI 자동 API 문서

## 알림 상태

`탕무스`를 내 캐릭터, `꽃게쥬`를 라이벌로 설정합니다. 비교 점수는 `레벨 × 100 + 현재 레벨 경험치 진행률`이며, 차이에 따라 아래 네 상태 중 하나를 저장합니다.

- 내가 1%p 이상 앞섬
- 내가 앞서지만 1%p 미만
- 라이벌이 앞서지만 1%p 미만
- 라이벌이 1%p 이상 앞섬

웹훅 URL이나 캐릭터 구성이 바뀌면 당시 상태를 새로운 기준으로 저장하며 최초 알림은 보내지 않습니다. 이후 서버가 실행 중인 동안 1시간마다 확인하여 상태가 변경된 경우에만 알립니다.

## 데이터 해석

현재 버전은 종합 랭킹 API의 `character_exp`를 표시합니다. 레벨업으로 경험치 값이 초기화된 구간은 잘못된 음수 성장량을 표시하지 않고 결측으로 처리합니다. 여러 레벨을 아우르는 완전한 누적 경험치는 레벨별 필요 경험치 테이블을 추가하는 다음 단계에서 지원합니다.

Data based on NEXON Open API
