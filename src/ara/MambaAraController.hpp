#pragma once

#include "ARA_Library/PlugIn/ARAPlug.h"
#include "../core/MambaInferenceEngine.hpp"
#include <memory>
#include <vector>
#include <atomic>

// Resolve model directory relative to the .so location (via dladdr), cached.
const std::string& getMambaModelDir();

class MambaPlaybackRenderer : public ARA::PlugIn::PlaybackRenderer
{
public:
    explicit MambaPlaybackRenderer(ARA::PlugIn::DocumentController* documentController) noexcept;
    ~MambaPlaybackRenderer() noexcept override;

protected:
    void willAddPlaybackRegion(ARA::PlugIn::PlaybackRegion* playbackRegion) noexcept override;
    void didAddPlaybackRegion(ARA::PlugIn::PlaybackRegion* playbackRegion) noexcept override;
    void willRemovePlaybackRegion(ARA::PlugIn::PlaybackRegion* playbackRegion) noexcept override;
    void didRemovePlaybackRegion(ARA::PlugIn::PlaybackRegion* playbackRegion) noexcept override;

private:
    std::unique_ptr<MambaInferenceEngine> rtEngine_;
    bool engineValid_{false};
};

class MambaDocumentController : public ARA::PlugIn::DocumentController
{
public:
    using ARA::PlugIn::DocumentController::DocumentController;

    // State params (mirrors CLAP MambaPresetData)
    std::atomic<float> bypass{0.0f};
    std::atomic<float> wetDry{1.0f};
    std::atomic<float> intensity{0.5f};

protected:
    ARA::PlugIn::Document* doCreateDocument() noexcept override;
    void doDestroyDocument(ARA::PlugIn::Document* document) noexcept override;

    ARA::PlugIn::MusicalContext* doCreateMusicalContext(ARA::PlugIn::Document* document, ARA::ARAMusicalContextHostRef hostRef) noexcept override;
    void doDestroyMusicalContext(ARA::PlugIn::MusicalContext* musicalContext) noexcept override;
    void doUpdateMusicalContextContent(ARA::PlugIn::MusicalContext* musicalContext, const ARA::ARAContentTimeRange* range, ARA::ContentUpdateScopes scopeFlags) noexcept override;

    ARA::PlugIn::RegionSequence* doCreateRegionSequence(ARA::PlugIn::Document* document, ARA::ARARegionSequenceHostRef hostRef) noexcept override;
    void doDestroyRegionSequence(ARA::PlugIn::RegionSequence* regionSequence) noexcept override;

    ARA::PlugIn::AudioSource* doCreateAudioSource(ARA::PlugIn::Document* document, ARA::ARAAudioSourceHostRef hostRef) noexcept override;
    void doDestroyAudioSource(ARA::PlugIn::AudioSource* audioSource) noexcept override;
    void doUpdateAudioSourceContent(ARA::PlugIn::AudioSource* audioSource, const ARA::ARAContentTimeRange* range, ARA::ContentUpdateScopes scopeFlags) noexcept override;
    void willEnableAudioSourceSamplesAccess(ARA::PlugIn::AudioSource* audioSource, bool enable) noexcept override;
    void didEnableAudioSourceSamplesAccess(ARA::PlugIn::AudioSource* audioSource, bool enable) noexcept override;

    ARA::PlugIn::AudioModification* doCreateAudioModification(ARA::PlugIn::AudioSource* audioSource, ARA::ARAAudioModificationHostRef hostRef, const ARA::PlugIn::AudioModification* optionalModificationToClone) noexcept override;
    void doDestroyAudioModification(ARA::PlugIn::AudioModification* audioModification) noexcept override;

    ARA::PlugIn::PlaybackRegion* doCreatePlaybackRegion(ARA::PlugIn::AudioModification* modification, ARA::ARAPlaybackRegionHostRef hostRef) noexcept override;
    void doDestroyPlaybackRegion(ARA::PlugIn::PlaybackRegion* playbackRegion) noexcept override;

    ARA::PlugIn::PlaybackRenderer* doCreatePlaybackRenderer() noexcept override;
    void doDestroyPlaybackRenderer(ARA::PlugIn::PlaybackRenderer* playbackRenderer) noexcept override;

    ARA::PlugIn::EditorRenderer* doCreateEditorRenderer() noexcept override;
    void doDestroyEditorRenderer(ARA::PlugIn::EditorRenderer* editorRenderer) noexcept override;

    ARA::PlugIn::EditorView* doCreateEditorView() noexcept override;
    void doDestroyEditorView(ARA::PlugIn::EditorView* editorView) noexcept override;

    bool doRestoreObjectsFromArchive(ARA::PlugIn::HostArchiveReader* archiveReader, const ARA::PlugIn::RestoreObjectsFilter* filter) noexcept override;
    bool doStoreObjectsToArchive(ARA::PlugIn::HostArchiveWriter* archiveWriter, const ARA::PlugIn::StoreObjectsFilter* filter) noexcept override;

private:
    bool audioSourceAccessEnabled_{false};
};

class MambaFactoryConfig : public ARA::PlugIn::FactoryConfig
{
public:
    ARA::PlugIn::DocumentController* createDocumentController(const ARA::PlugIn::PlugInEntry* entry, const ARA::ARADocumentControllerHostInstance* instance) const noexcept override;

    const char* getFactoryID() const noexcept override;
    const char* getPlugInName() const noexcept override;
    const char* getManufacturerName() const noexcept override;
    const char* getInformationURL() const noexcept override;
    const char* getVersion() const noexcept override;
    const char* getDocumentArchiveID() const noexcept override;
};

__attribute__((visibility("default"))) const ARA::ARAFactory* getMambaARAFactory() noexcept;

extern "C" __attribute__((visibility("default"))) const ARA::ARAFactory* ARAGetFactory();
