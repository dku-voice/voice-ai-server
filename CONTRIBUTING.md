# 브랜치 전략

## 브랜치 구조
- main: 배포 브랜치
- develop: 개발 통합 브랜치
- feature/기능명: 기능 개발 브랜치

## 브랜치 규칙
1. main에 직접 push 금지
2. 모든 작업은 feature 브랜치에서 시작
3. develop으로 PR 후 머지
4. develop → main PR은 PM 김진우 승인 필수

## 커밋 메시지 규칙
- feat: 새로운 기능
- fix: 버그 수정
- ci: CI/CD 관련
- docs: 문서 수정
- refactor: 리팩토링
- test: 테스트 코드

## PR 규칙
1. PR 제목은 커밋 메시지 규칙 따르기
2. PR 설명에 작업 내용 간단히 작성
3. 충돌 발생 시 PM에게 보고
